#!/usr/bin/env python3
"""Evaluate frozen h28 language prompt plus raw l28 positive point prompts."""
from __future__ import annotations
import argparse, gc, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tools.multilayer_seg import evaluate_raw_similarity_box_prompt as box
from tools.multilayer_seg.evaluate_spatial_map_affine_gate import build_event, infer_baseline_logits
from tools.multilayer_seg.extract_spatial_adapter_features import extract_states
from tools.multilayer_seg.run_dynamic_k_oracle import exact_metrics
from tools.multilayer_seg.train_eval_fusion_standard import cache_video_backbones, ground_truth_video, preprocess_video

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('model_path'); p.add_argument('--data-root',required=True); p.add_argument('--manifest',required=True); p.add_argument('--raw-map-dir',required=True); p.add_argument('--output',required=True); p.add_argument('--split',choices=('calibration',),default='calibration'); p.add_argument('--point-counts',type=int,nargs='+',default=(1,3,5)); p.add_argument('--nms-radius',type=int,default=2); p.add_argument('--max-samples',type=int,default=8); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--num-shards',type=int,default=1); return p.parse_args()

def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)); tmp.replace(path)

def protocol_hash(manifest, counts, radius):
    payload={'split_hash':manifest['calibration_sha256'],'map_layer':28,'language_layer':28,'conditioning_frames':5,'point_counts':counts,'nms_radius':radius,'point_source':'raw_l28_similarity','propagation':'complete_video','training':'none'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def select_points(score, count, radius, image_size, device):
    a=score.float().cpu().numpy(); side=a.shape[0]; order=np.argsort(-a.reshape(-1),kind='mergesort'); chosen=[]
    for flat in order:
        y,x=divmod(int(flat),side)
        if all((x-cx)**2+(y-cy)**2>radius**2 for cy,cx in chosen): chosen.append((y,x))
        if len(chosen)>=count: break
    scale=float(image_size)/side
    coords=torch.tensor([[[x*scale+scale/2,y*scale+scale/2] for y,x in chosen]],dtype=torch.float32,device=device).clamp(0.,float(image_size)-.5)
    labels=torch.ones((1,len(chosen)),dtype=torch.int32,device=device)
    return {'point_coords':coords,'point_labels':labels}, [{'x':int(x),'y':int(y),'score':float(a[y,x])} for y,x in chosen]

@torch.inference_mode()
def infer_points(grounding,images,cache,embedding,raw_maps,output_size,count,radius):
    predictor=grounding.sam2_model; state=predictor.init_state(images); state['cached_features']=cache.copy(); details=[]; obj_id=100; obj_idx=None
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        for frame in range(min(5,len(images),len(raw_maps))):
            points, selected=select_points(raw_maps[frame],count,radius,int(predictor.image_size),embedding.device); details.append({'frame_index':frame,'points':selected})
            if obj_idx is None: obj_idx=box.register_object(state,obj_id)
            out=state['output_dict_per_obj'][obj_idx]; temp=state['temp_output_dict_per_obj'][obj_idx]; init=frame not in state['frames_already_tracked']; key='cond_frame_outputs' if (init or predictor.add_all_frames_to_correct_as_cond) else 'non_cond_frame_outputs'
            language=embedding.reshape(1,1,-1)
            current,_=predictor._run_single_frame_inference(inference_state=state,output_dict=out,frame_idx=frame,batch_size=1,is_init_cond_frame=init,point_inputs=points,mask_inputs=None,reverse=False,run_mem_encoder=False,prev_sam_mask_logits=None,language_embd=language)
            temp[key][frame]=current; state['point_inputs_per_obj'][obj_idx][frame]={k:v.to(state['storage_device'],non_blocking=True) for k,v in points.items()}
        by_frame={int(i):logits for i,_,logits in predictor.propagate_in_video(state)}; logits=torch.cat([by_frame[i] for i in range(len(images))])
    w,h=output_size; logits=F.interpolate(logits.float(),size=(h,w),mode='bilinear',align_corners=False)[:,0].cpu(); return logits,details

def main():
    a=parse_args(); counts=tuple(a.point_counts)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if any(c<1 for c in counts) or a.nms_radius<0: raise ValueError('invalid point settings')
    out=Path(a.output); rows=out/'rows'; rows.mkdir(parents=True,exist_ok=True); manifest=json.loads(Path(a.manifest).read_text()); selected=[x for i,x in enumerate(manifest[a.split][:a.max_samples]) if i%a.num_shards==a.shard_index]; ph=protocol_hash(manifest,counts,a.nms_radius); atomic(out/'status.json',{'state':'running','phase':'model_load','protocol_hash':ph,'samples':len(selected)})
    model=AutoModel.from_pretrained(a.model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True).eval().to('cuda:0'); tok=AutoTokenizer.from_pretrained(a.model_path,trust_remote_code=True); model.preparing_for_generation(tokenizer=tok); [p.requires_grad_(False) for p in model.parameters()]; grounding=model.grounding_encoder; projection=model.text_hidden_fcs; dtype=next(projection.parameters()).dtype; root=Path(a.data_root); split=root/'video_datas'/'mevis'/'train'; metadata=json.loads((split/'meta_expressions.json').read_text())['videos']; masks=json.loads((split/'mask_dict.json').read_text()); done=0
    for n,item in enumerate(selected,1):
        event=build_event(root,item,metadata); payload=torch.load(Path(a.raw_map_dir)/(event['key']+'.pt'),map_location='cpu',weights_only=False); layers=[int(x) for x in payload['layers']]; raw=payload['maps'][layers.index(28)].float(); seg,_,valid,side=extract_states(model,tok,event,5,(28,)); emb=projection(seg[0].to('cuda:0',dtype=dtype)).reshape(-1); images,size=preprocess_video(grounding,model.extra_image_processor,model.torch_dtype,event['frame_paths']); w,h=size; truth=ground_truth_video(masks,event['annotation_ids'],len(images),(h,w)); cache=cache_video_backbones(grounding,images); base=infer_baseline_logits(grounding,images,cache,emb,size); bm=exact_metrics(truth,base.numpy()>0); candidates=[]
        for c in counts:
            logits,details=infer_points(grounding,images,cache,emb,raw,size,c,a.nms_radius); m=exact_metrics(truth,logits.numpy()>0); candidates.append({'candidate_id':f'seg_points_top{c}','point_count':c,'metrics':m,'delta_J&F_points':(m['J&F']-bm['J&F'])*100,'conditioning_points':details}); del logits
        atomic(rows/(event['key']+'.json'),{'key':event['key'],'video_id':event['video_id'],'exp_id':event['exp_id'],'split':a.split,'protocol_hash':ph,'baseline_h28':bm,'candidates':candidates,'contract':'frozen h28 language + raw l28 spatially separated positive points; no training; complete-video propagation'}); done+=1; print(f"RAW_POINTS [{n}/{len(selected)}] {event['key']} baseline={bm['J&F']:.4f} "+' '.join(f"top{x['point_count']}={x['delta_J&F_points']:+.3f}pt" for x in candidates),flush=True); del payload,raw,seg,emb,images,cache,base,truth; gc.collect(); torch.cuda.empty_cache()
    atomic(out/'status.json',{'state':'complete','phase':'done','completed':done,'total':len(selected),'protocol_hash':ph})
if __name__=='__main__': main()
