import {PointInspector} from './inspection.js';
import {renderGuide, renderWorkflow} from './handeye_guide.js';
import {animatePosePreview} from './handeye_direction.js';
const $ = id => document.getElementById(id);
const text = (id, value) => {if ($(id).textContent !== String(value)) $(id).textContent = value;};
const visible = (id, value) => {$(id).hidden = !value;};
const fmt = (value, n=1) => Number.isFinite(value) ? value.toFixed(n) : '—';
let stereo, state, busy=false, runs=[], selectedKey='', after=-1, blobUrl, lastVideo=0, videoTimes=[];
let measuredUrl, measuredKey='', videoRevision=0, videoAbort, measuredTime=0;
let captureKey='', reportKey='', runtimeError='';
let displayed = null;
const inspector = new PointInspector({sourceId:'source',boardUrl:'/api/handeye/board.png',pointRadius:8,overlayId:'measuredOverlay',selectMeasuredOnIds:false});
function inspectionState(s, m, cameraState) {
    return {inspection:m.inspection,frame_id:m.frame_id,workspace_id:s.workspace_id + ':' + s.run_id,
            camera:cameraState.camera,board:s.board,has_model:true,candidate:cameraState.candidate};
}
function updateInspection(){
    const measured=$('source').value==='measured';
    visible('measuredPanel',!measured);
    inspector.setOverlay(measured?'pointOverlay':'measuredOverlay');
    if(displayed && displayed.run_id===state?.run_id && displayed.workspace_id===state?.workspace_id)
        inspector.update(displayed.inspection,'raw',true);
    else inspector.clear('Waiting for a fresh detected frame');
}
function showMeasuredMain(){
    if(!displayed||$('source').value!=='measured')return;
    $('frame').src=measuredUrl;visible('frame',true);visible('emptyFrame',false);
    $('frame').dataset.frameId=displayed.frame_id;$('frame').dataset.source='measured';
    text('videoLabel','Inspection image · updates at detection speed');
    text('videoRate',`Frame ${displayed.frame_id} · slower than live video`);
}
async function measuredFrame(){
    if(!state?.active||state.workspace_id!==stereo?.workspace_id||document.hidden)return;
    if(performance.now()-measuredTime<500)return;
    const s=state,camera=stereo,m=s.measurement;if(!m||m.age_s>=3)return;
    const key=`${s.workspace_id}:${s.run_id}:${m.frame_id}`;if(key===measuredKey)return;
    const q=new URLSearchParams({workspace_id:s.workspace_id,frame_id:m.frame_id});
    const response=await fetch(`/api/handeye/frame.jpg?${q}`);
    if(!response.ok)return;
    const url=URL.createObjectURL(await response.blob()),img=new Image();img.src=url;
    try{await img.decode();}catch(e){URL.revokeObjectURL(url);throw e;}
    if(!state.active||s.workspace_id!==state.workspace_id||s.run_id!==state.run_id||Number(response.headers.get('X-Frame-Id'))!==m.frame_id){URL.revokeObjectURL(url);return;}
    const old=measuredUrl;measuredUrl=url;measuredKey=key;measuredTime=performance.now();
    $('measuredFrame').src=url;visible('measuredFrame',true);$('measuredFrame').dataset.frameId=m.frame_id;
    displayed={run_id:s.run_id,workspace_id:s.workspace_id,frame_id:m.frame_id,inspection:inspectionState(s,m,camera)};
    text('measuredRate',`Frame ${m.frame_id} · up to 2 updates/s`);
    showMeasuredMain();updateInspection();if(old)URL.revokeObjectURL(old);
}
async function request(path, body) {
    const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,workspace_id:stereo?.workspace_id})} : {});
    if (!response.ok) {const e=await response.json().catch(()=>({}));throw Error(typeof e.detail==='string'?e.detail:`Request failed (${response.status})`);}
    return response.json();
}
function error(message) {text('error',message);visible('error',!!message);}
function render() {
    if (!stereo || !state) return;
    text('connection',stereo.stream_only?'Stream only':'Connected');
    text('streamOnly',stereo.stream_only?'Resume board detection':'Stream only');
    $('streamOnly').disabled=busy || stereo.phase==='picker';
    const current=state.workspace_id===stereo.workspace_id;
    const working=current && state.active;
    const stage=working?state.stage:'setup';
    const visibleStage=['training','checking'].includes(stage)?'guided':stage;
    document.querySelectorAll('[data-stage]').forEach(e=>e.classList.toggle('active',e.dataset.stage===visibleStage));
    visible('setup',!working);visible('working',working);
    visible('newRun',working);
    $('newRun').disabled=busy || state.busy || state.pending || state.assessing;
    $('folder').disabled=!current || !state.path || busy;
    text('calibration',stereo.has_model?`Stereo session: ${stereo.session.split('/').pop()} · ${stereo.camera.width/2} × ${stereo.camera.height} per eye`:'Open a completed stereo session first using Stereo calibration above.');
    $('start').disabled=busy || !$('mounted').checked || !stereo.has_model || stereo.replay;
    $('resume').disabled=busy || !$('mounted').checked || !$('runs').value || stereo.replay;
    if (!working) return;
    const ranges=Object.entries(state.kinematic_calibration?.joint_angle_maps||{});
    visible('modelRange',ranges.length>0);
    text('modelRange',ranges.map(([joint,spec])=>`${joint.replace(/^joint/,'J')}: ${spec.qualified_range_rad.map(v=>fmt(v*180/Math.PI,1)+'°').join(' to ')}`).join(' · ') + (ranges.length?(state.kinematic_calibration?.configuration_correction?' — angle-map domain.':' — measured range validated for this software correction.'):'') + (state.kinematic_calibration?.configuration_correction?` Static compensation passed ${state.kinematic_calibration.qualification.independent_views} fresh stationary poses. Keep the same payload and mounting; other poses and physical task accuracy need validation.`:''));
    const checking=stage==='checking', complete=stage==='complete';
    renderWorkflow(state);
    $('capture').disabled=busy || state.auto_capture || !state.capture_ready;visible('capture',!complete);
    text('capture',state.pending?'Capturing…':state.just_captured?'✓ Captured — move to next pose':'Capture this pose manually');
    $('capture').dataset.pending=String(state.pending);
    $('undo').disabled=busy || !state.training_count || state.pending;visible('undo',!checking&&!complete);
    $('pause').disabled=busy;text('pause',complete?'Finish session':'Pause session');
    renderGuide(state,busy);
    const m=state.measurement, fresh=m && m.age_s<3;
    updateInspection();
    const limits=fresh?(m.joint_limit_diagnostics||[]):[];
    visible('jointNotice',limits.length>0);
    text('jointNotice',limits.map(d=>`${d.joint}: measured ${fmt(d.measured_rad,3)} rad; nominal command range ${fmt(d.lower_rad,3)} to ${fmt(d.upper_rad,3)} rad.`).join(' ') + (limits.length?' Calibration records the measured angle unchanged. Robot command limits are unchanged.':''));
    text('feedback',state.feedback.connected?`Live · ${fmt(state.feedback.age_s*1000,0)} ms`:'No fresh feedback');
    $('feedback').className=state.feedback.connected?'good':'bad';text('topic',state.feedback.error||state.feedback.topic||'');
    text('skew',fresh&&m.pair_skew_ms!=null?`${fmt(m.pair_skew_ms)} ms`:'—');
    text('reprojection',fresh&&m.reprojection_px?`${fmt(m.reprojection_px[0],2)} / ${fmt(m.reprojection_px[1],2)} px`:'—');
    text('path',state.path||'');
    const captures=JSON.stringify(state.captures);
    if(captures!==captureKey){
        captureKey=captures;$('captures').replaceChildren();$('excludedCaptures').replaceChildren();
        const excluded=state.captures.filter(c=>c.excluded);
        visible('excludedDetails',excluded.length>0);text('excludedCount',`${excluded.length} excluded poses`);
        for(const c of state.captures){const e=document.createElement('span');e.textContent=`${c.role==='check'?'Validation':'Fit'} ${c.id.split('_')[0]}`;e.title=c.id;(c.excluded?$('excludedCaptures'):$('captures')).append(e);}
    }
    visible('result',!!state.solution);
    if(state.solution){
        text('resultTitle',complete?'Camera-to-arm transform saved':'Fixed candidate · validation');
        const checks=state.report?.checks||[], failures=checks.filter(c=>!c.passed).length;
        text('resultBadge',complete?'Passed':failures?'Needs review':'Collecting evidence');
        text('resultText',complete?'Independent poses agree with the frozen fit. Download the transform and retain this session for provenance.':failures?'Fresh poses disagree with the fixed candidate. Their results are saved separately from the fitting poses. Review the images, board mounting and measured arm poses before continuing.':`Keep using guided capture across new wrist rotations and positions. Each fresh pose must stay within ${fmt(state.readiness?.threshold_px??3,1)} px at the original camera resolution; coverage must also qualify. Physical accuracy needs a separate check.`);
        const key=JSON.stringify(checks);
        if(key!==reportKey){reportKey=key;$('checks').replaceChildren();for(const c of checks){const tr=document.createElement('tr');for(const value of [c.id.replaceAll('_',' '),`${fmt(c.left_rms_px,2)} / ${fmt(c.right_rms_px,2)}`,`${fmt(c.translation_mm)} mm / ${fmt(c.rotation_deg)}°`,c.passed?'Pass':c.error||'Needs review']){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('checks').append(tr);}}
        const query=new URLSearchParams({workspace_id:stereo.workspace_id,run_id:state.run_id});
        for(const [id,file,show] of [['downloadReport','result.json',!!state.report],['downloadTransform','handeye.npz',complete],['downloadModel',state.arm_model_file||'arm_model.urdf',complete],['downloadRun','run.json',true]]){$(id).href=`/api/handeye/artifacts/${file}?${query}`;visible(id,show);}
    }
    if(state.error){error(state.error);runtimeError=state.error;}
    else if(runtimeError){if($('error').textContent===runtimeError)error('');runtimeError='';}
}
async function refreshRuns(){
    runs=await request('/api/handeye/runs');const previous=$('runs').value;$('runs').replaceChildren(new Option('Saved hand–eye sessions',''));
    for(const r of runs)$('runs').append(new Option(`${r.id} · ${r.side} · ${r.samples} poses`,r.id));
    if(runs.some(r=>r.id===previous))$('runs').value=previous;
}
async function action(fn){busy=true;render();error('');try{await fn();await refreshRuns();}catch(e){error(e.message);}finally{busy=false;render();}}
$('streamOnly').onclick=()=>action(async()=>{
    stereo=await request('/api/control',{action:stereo.stream_only?'resume_detection':'stream_only'});
    state=await request('/api/handeye/state');
    if(stereo.stream_only){$('source').value='live';displayed=null;updateInspection();}
});
$('autoCapture').onclick=()=>action(async()=>{state=await request('/api/handeye/control',{action:state.auto_capture?'auto_stop':'auto_start'});});
$('reviewDetails').onclick=()=>{
    const details=$('calibrationDetails');details.open=true;
    details.querySelector('summary').focus({preventScroll:true});
    details.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
};
$('retryCapture').onclick=()=>action(async()=>{state=await request('/api/handeye/control',{action:'retry_capture'});});
$('excludeFailed').onclick=()=>action(async()=>{state=await request('/api/handeye/control',{action:'exclude_failed_training'});});
$('guideNext').onclick=()=>action(async()=>{state=await request('/api/handeye/control',{action:'guide_next'});});
$('newRun').onclick=()=>action(async()=>{
    const side=state.side;
    state=await request('/api/handeye/control',{action:'pause'});
    $('side').value=side;
    $('mounted').checked=false;
    $('runs').value='';
    render();
    $('mounted').focus({preventScroll:true});
    $('setup').scrollIntoView({block:'start'});
});
$('start').onclick=()=>action(async()=>{state=await request('/api/handeye/start',{side:$('side').value});});
$('resume').onclick=()=>action(async()=>{const r=runs.find(r=>r.id===$('runs').value);state=await request('/api/handeye/start',{side:r.side,run_id:r.id});});
for(const id of ['capture','undo','pause','folder'])$(id).onclick=()=>action(async()=>{state=await request('/api/handeye/control',{action:id});});
for(const id of ['mounted','side','runs'])$(id).onchange=render;
$('source').onchange=()=>{
    videoRevision++;videoAbort?.abort();after=-1;lastVideo=0;videoTimes=[];
    if($('source').value==='measured')showMeasuredMain();
    updateInspection();
};
async function poll(){try{
    [stereo,state]=await Promise.all([request('/api/state'),request('/api/handeye/state')]);
    if(selectedKey!==stereo.workspace_id){
        selectedKey=stereo.workspace_id;await refreshRuns();after=-1;measuredKey='';displayed=null;
        videoRevision++;videoAbort?.abort();
    }
    render();await measuredFrame();
    if(measuredTime&&performance.now()-measuredTime>2500){
        displayed=null;updateInspection();text('measuredRate','Detection paused · waiting for a fresh measurement');
    }
}catch(e){text('connection','Disconnected');error(e.message);}setTimeout(poll,200);}
async function video(){const start=performance.now();try{
    // Same latest-only live endpoint as stereo mode. Measurement fetches never block this loop.
    if(state?.active&&state.workspace_id===stereo?.workspace_id&&!document.hidden&&$('source').value==='live'){
        const revision=videoRevision,s=state;
        const q=new URLSearchParams({workspace_id:s.workspace_id,mode:'raw',after});
        videoAbort=new AbortController();
        const response=await fetch(`/api/live.jpg?${q}`,{signal:videoAbort.signal});
        if(response.ok&&response.status!==204){
            const url=URL.createObjectURL(await response.blob()),img=new Image();img.src=url;
            try{await img.decode();}catch(e){URL.revokeObjectURL(url);throw e;}
            if(revision===videoRevision&&$('source').value==='live'&&state.active&&state.workspace_id===s.workspace_id&&state.run_id===s.run_id){
                $('frame').src=url;visible('frame',true);visible('emptyFrame',false);
                if(blobUrl)URL.revokeObjectURL(blobUrl);blobUrl=url;after=Number(response.headers.get('X-Frame-Id'));
                $('frame').dataset.frameId=after;$('frame').dataset.source='live';
                $('frame').alt='Live stereo camera';
                lastVideo=performance.now();videoTimes.push(lastVideo);videoTimes=videoTimes.filter(t=>t>lastVideo-2000);
                const fps=videoTimes.length>1?(videoTimes.length-1)*1000/(lastVideo-videoTimes[0]):0;
                text('videoRate',`${fmt(fps)} fps · live`);
                $('videoRate').title=`${response.headers.get('X-Frame-Age-Ms')} ms from camera receipt to preview encoding`;
                text('videoLabel','Live stereo');
            }else URL.revokeObjectURL(url);
        }
    }
}catch(e){if(e.name!=='AbortError')text('videoRate',e.message);}
if($('source').value==='live'&&lastVideo&&performance.now()-lastVideo>2500)text('videoRate','Waiting for live video');
setTimeout(video,Math.max(0,1000/30-(performance.now()-start)));}
function animate(now){
    if(!document.hidden&&state?.active)animatePosePreview(now);
    requestAnimationFrame(animate);
}
poll();video();requestAnimationFrame(animate);
