import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {SettledStatus, renderGuide, renderWorkflow} from '../stereo_calibration/static/handeye_guide.js';
import {updateMovementGuide, updateBoardGraphic} from '../stereo_calibration/static/movement_guide.js';

const I=[[1,0,0],[0,1,0],[0,0,1]];
function nodesFor(page){
    const html=readFileSync(new URL(`../stereo_calibration/static/${page}`,import.meta.url),'utf8');
    const nodes=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(([,id])=>{
        const n={textContent:'',hidden:false,disabled:false,attributes:{},dataset:{},classes:new Map(),style:{}};
        n.classList={toggle:(key,on)=>n.classes.set(key,on)};
        n.setAttribute=(key,value)=>n.attributes[key]=String(value);
        n.getAttribute=key=>n.attributes[key];
        Object.defineProperty(n,'innerHTML',{set(){throw Error('Animation DOM must not be rebuilt');}});
        return [id,n];
    }));
    globalThis.document={getElementById(id){assert.ok(nodes.has(id),`Missing UI element: ${id}`);return nodes.get(id);}};
    return nodes;
}

test('status does not oscillate between ready and checking on alternating quality frames',()=>{
    const s=new SettledStatus();
    for(let i=0;i<8;i++){
        assert.equal(s.update(i%2===0,i%2===0?'Ready':'Board reprojection is unstable.',i*500),
                     'Waiting for a stable measurement.');
        assert.equal(s.good,false);
    }
    s.update(true,'Ready',4000);
    assert.equal(s.update(true,'Ready',5000),'Ready');
    assert.equal(s.good,true);
    assert.equal(s.update(false,'Corners missing',5100),'Waiting for a stable measurement.');
    assert.equal(s.good,false);
    assert.equal(s.update(false,'Corners missing',6100),'Corners missing');
});

test('numeric blocker readings settle by failure type instead of restarting on every value',()=>{
    const s=new SettledStatus();
    s.update(false,'Joint varied 0.6°',0,'motion');
    s.update(false,'Joint varied 0.7°',500,'motion');
    assert.equal(s.update(false,'Joint varied 0.8°',1000,'motion'),'Joint varied 0.8°');
    assert.equal(s.good,false);
});

test('valid pose can start recording away from the optional orientation and blue circle',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'anywhere',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:true,hold_progress:.5,
        capture_ready:true,just_captured:false,message:'Ready',measurement:{frame_id:1,age_s:0,reason:'Ready'},
        guidance:{key:'tilt-away',title:'Move into the blue area',instruction:'Move centre',
            current_rotation:I,target_rotation:I,angle_error_deg:30,at_target:false,
            axis_span_deg:[0,0,0],position_radius:.14,position_target:[.5,.5],
            adjustment:{kind:'move',target:[.5,.5],current:[.8,.5]}}};
    renderGuide(state,false,1000);
    assert.equal(nodes.get('holdLabel').textContent,'Valid new pose — hold still to record');
    assert.equal(nodes.get('guideTitle').textContent,'Good pose — hold still');
    assert.equal(nodes.get('holdPercent').textContent,'50%');
    assert.equal(nodes.get('guideError').textContent,'Bring the board centre towards the blue dot');
});

test('same movement renderer handles stereo position, four printed edges, distance and hold',()=>{
    const nodes=nodesFor('index.html');
    const board={columns:10,rows:7,square_mm:20};updateBoardGraphic(board);
    updateMovementGuide({kind:'move',target:[.8,.2],current:[.2,.8]},board);
    assert.equal(nodes.get('illustration').dataset.motion,'position');
    assert.equal(nodes.get('boardDrawing').style.display,'none');
    assert.equal(Number(nodes.get('positionTarget').attributes.cx),38+.8*184);
    assert.equal(nodes.get('positionArrow').style.display,'');
    for(const edge of ['left','right','top','bottom']){
        updateMovementGuide({kind:'tilt',edge},board);
        assert.equal(nodes.get('illustration').dataset.motion,edge);
        assert.match(nodes.get('edgeHighlight').attributes.d,/^M/);
        assert.equal(nodes.get('positionDrawing').style.display,'none');
    }
    updateMovementGuide({kind:'distance',direction:'closer'},board,{holding:true});
    assert.equal(nodes.get('illustration').dataset.motion,'near');
    assert.equal(nodes.get('illustration').classes.get('holding'),true);
});

test('failed calibration checks do not hide the live capture readiness or blocker',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'check-readiness',run_id:'r',stage:'checking',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,hold_progress:0,
        report:{checks:[{passed:false},{passed:false}]},capture_ready:false,
        measurement:{frame_id:1,age_s:0,reason:'Need 12 shared corners; seeing 7.',blocker_code:'visibility'},
        guidance:{key:'manual',title:'Check pose',instruction:'Show board',current_rotation:I,target_rotation:I,
            axis_span_deg:[0,0,0],adjustment:{kind:'hold'}}};
    renderGuide(state,false,10000);renderGuide(state,false,11000);
    assert.equal(nodes.get('message').textContent,state.measurement.reason);
    assert.equal(nodes.get('capture').disabled,true);
    assert.equal(nodes.get('checkNotice').hidden,false);
    assert.match(nodes.get('checkNotice').textContent,/2 of 2 fresh poses failed/);
    state.capture_ready=true;state.measurement.reason='Ready to capture this steady pose.';
    renderGuide(state,false,12000);renderGuide(state,false,13000);
    assert.equal(nodes.get('message').textContent,state.measurement.reason);
    assert.equal(nodes.get('capture').disabled,false);
    assert.match(nodes.get('checkNotice').textContent,/recording is stopped/);
    state.report.checks=[{passed:true}];renderGuide(state,false,14000);
    assert.equal(nodes.get('checkNotice').hidden,true);
});

test('hand-eye target remains visible in left view through position, tilt, hold and dropout',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'w',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:true,hold_progress:0,
        capture_ready:false,just_captured:false,message:'Waiting',measurement:{frame_id:1,reason:'Waiting'},
        guidance:{key:'tilt-right',title:'Move into the blue area',instruction:'Move centre',
            current_rotation:I,target_rotation:I,target_vector_deg:[0,20,0],
            angle_error_deg:20,at_target:false,axis_span_deg:[0,0,0],position_radius:.14,position_target:[.5,.5],
            adjustment:{kind:'move',target:[.5,.5],current:[.8,.5]}}};
    renderGuide(state,false,1000);
    const illustration=nodes.get('illustration'),drawing=nodes.get('boardDrawing');
    assert.equal(illustration.dataset.motion,'position');
    assert.equal(nodes.get('positionOverlay').hidden,false);
    assert.equal(nodes.get('livePositionTarget').attributes.cx,'50');
    assert.equal(nodes.get('livePositionTarget').attributes.visibility,'visible');
    assert.equal(nodes.get('positionOverlay').dataset.mode,'target');
    assert.equal(nodes.get('positionLegendText').textContent,'Blue dot: board position');
    assert.ok(Math.abs(Number(nodes.get('livePositionTarget').attributes.rx)-14)<1e-12);
    assert.ok(Number(nodes.get('livePositionTarget').attributes.cx)+Number(nodes.get('livePositionTarget').attributes.rx)<100);
    state.guidance.adjustment={kind:'tilt',edge:'right'};
    state.guidance.title='Tilt right edge';renderGuide(state,false,1500);
    assert.equal(nodes.get('positionOverlay').hidden,false);
    assert.equal(nodes.get('positionTargetDot').style.left,'25%');
    assert.equal(nodes.get('positionTargetDot').style.top,'50%');
    assert.equal(nodes.get('positionLegend').hidden,false);
    assert.equal(illustration.dataset.motion,'right');
    assert.equal(drawing.style.display,'');
    state.guidance.current_rotation=null;
    for(let i=0;i<6;i++){
        renderGuide(state,false,2000+i*500);
        assert.equal(illustration.dataset.motion,'right');
        assert.equal(nodes.get('boardDrawing'),drawing);
        assert.equal(nodes.get('positionOverlay').hidden,false);
        assert.equal(illustration.classes.get('holding'),false);
    }
    state.hold_progress=.5;state.guidance.angle_error_deg=0;renderGuide(state,false,6000);
    assert.equal(illustration.classes.get('holding'),true);
    assert.equal(nodes.get('positionOverlay').hidden,false);
    state.hold_progress=0;state.guidance.angle_error_deg=null;renderGuide(state,false,6500);
    assert.equal(illustration.classes.get('holding'),true); // Brief dropout does not flash.
    renderGuide(state,false,7000);
    assert.equal(illustration.classes.get('holding'),false);
    state.guidance.key='done';renderGuide(state,false,8000);
    assert.equal(nodes.get('positionOverlay').hidden,true);
    assert.equal(nodes.get('positionLegend').hidden,true);
});

test('pose sequence holds red for 2 seconds, interpolates smoothly, then holds green',async()=>{
    const {PoseSequence,rotation,POSE_RED,POSE_GREEN}=await import('../stereo_calibration/static/handeye_direction.js');
    const target=rotation([Math.cos(Math.PI/12),0,Math.sin(Math.PI/12),0]);
    const seq=new PoseSequence();
    const feed=(at,current=I)=>seq.update({key:'a',current,target,observedAt:at},at);
    feed(0);assert.deepEqual(seq.sample(0).rotation,I);
    feed(1999);assert.equal(seq.sample(1999).phase,'current');
    assert.deepEqual(seq.sample(1999).colour,POSE_RED);
    feed(3000);const mid=seq.sample(3000);
    assert.equal(mid.phase,'moving');assert.equal(mid.progress,.5);
    assert.ok(Math.abs(mid.rotation[0][0]-Math.cos(Math.PI/12))<1e-12);
    assert.notDeepEqual(mid.colour,POSE_RED);assert.notDeepEqual(mid.colour,POSE_GREEN);
    feed(4000);assert.equal(seq.sample(4000).phase,'target');
    assert.deepEqual(seq.sample(4000).colour,POSE_GREEN);
    for(let r=0;r<3;r++)for(let c=0;c<3;c++)assert.ok(Math.abs(seq.sample(4000).rotation[r][c]-target[r][c])<1e-12);
    feed(5900,target);assert.equal(seq.sample(5900).phase,'target');
    assert.equal(seq.sample(6000).phase,'current'); // One new snapshot per cycle, not per poll.
    assert.ok(Math.abs(seq.sample(6000).rotation[0][2]-target[0][2])<1e-12);
    assert.equal(seq.sample(8900).phase,'waiting'); // No endless animation of an old observation.
});

test('camera coordinates preserve sideways boards and depth cue distinguishes toward / away',async()=>{
    const {rotation,quaternion,slerp,cameraToCss,depthCue}=await import('../stereo_calibration/static/handeye_direction.js');
    const quarter=rotation([Math.SQRT1_2,0,0,-Math.SQRT1_2]);
    const css=cameraToCss(quarter).slice(9,-1).split(',').map(Number);
    assert.ok(Math.abs(css[0])<1e-12);assert.ok(Math.abs(css[1]+1)<1e-12);
    for(const sign of [1,-1]){
        const tilted=rotation([Math.cos(Math.PI/12),0,sign*Math.sin(Math.PI/12),0]);
        const cue=depthCue(I,tilted);
        assert.equal(cue.edge,'right');assert.equal(cue.direction,sign>0?'toward':'away');
        // CSS z has the opposite sign from camera z, so a nearer right edge grows.
        const matrix=cameraToCss(tilted).slice(9,-1).split(',').map(Number);
        assert.equal(Math.sign(matrix[2]),sign);
    }
    assert.equal(depthCue(I,quarter).edge,null); // In-plane turn must not invent a depth cue.
    const tilted=rotation([Math.cos(.4),0,Math.sin(.4),0]);
    const turned=tilted.map(row=>quarter[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*quarter[k][j],0)));
    assert.equal(depthCue(tilted,turned).edge,null);
    assert.match(depthCue(tilted,turned).label,/counterclockwise/);
    for(const axis of [0,1,2]){
        const q=[0,0,0,0];q[axis+1]=1;
        const round=rotation(quaternion(rotation(q)));
        assert.deepEqual(round,rotation(q)); // Including 180-degree branches.
    }
    const q=quaternion(quarter);
    assert.ok(slerp(q,q.map(v=>-v),.5).every((v,i)=>Math.abs(v-q[i])<1e-12));
});

test('target change resets demonstration; alignment and reduced motion remain explicit',async()=>{
    const {PoseSequence,rotation}=await import('../stereo_calibration/static/handeye_direction.js');
    const seq=new PoseSequence(),target=rotation([Math.cos(.2),Math.sin(.2),0,0]);
    seq.update({key:'one',current:I,target,observedAt:0},0);seq.sample(0);
    seq.update({key:'one',current:I,target,observedAt:2200},2200);
    assert.equal(seq.sample(2200,true).phase,'moving');
    for(let r=0;r<3;r++)for(let c=0;c<3;c++)assert.ok(Math.abs(seq.sample(2200,true).rotation[r][c]-target[r][c])<1e-12);
    seq.update({key:'two',current:I,target,observedAt:2400},2400);
    assert.equal(seq.sample(2400).phase,'current');
    seq.update({key:'two',current:target,target,observedAt:2500,aligned:true},2500);
    assert.equal(seq.sample(2500).phase,'aligned');
    assert.equal(seq.sample(5500).phase,'waiting');
    seq.update({key:'three',current:I,target,observedAt:6000,captureState:'recording'},6000);
    assert.equal(seq.sample(6000).phase,'recording');
    assert.deepEqual(seq.sample(6000).rotation,I); // Show actual pose while recording, not suggestion.
});

test('both tilt orders reconstruct arbitrary board poses, including gimbal lock',async()=>{
    const {axisPlan,multiply,axisRotation,rotation}=await import('../stereo_calibration/static/handeye_direction.js');
    const near=(a,b,tol=1e-10)=>a.forEach((row,i)=>row.forEach((v,j)=>assert.ok(Math.abs(v-b[i][j])<tol)));
    let seed=17;
    const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
    for(const order of ['YXZ','XYZ'])for(let i=0;i<200;i++){
        const q=Array.from({length:4},()=>rand()-.5),norm=Math.hypot(...q);
        const current=rotation(q.map(v=>v/norm));
        const middle=i<4?(i%2?1:-1)*(Math.PI/2-(i<2?0:1e-8)):(rand()-.5)*2*Math.PI;
        const target=multiply(multiply(multiply(current,axisRotation(order[0],(rand()-.5)*6)),axisRotation(order[1],middle)),axisRotation('Z',(rand()-.5)*6));
        const steps=axisPlan(current,target,.8,order);
        assert.deepEqual(steps.map(s=>s.axis),[...order]);
        near(steps[0].from,current);near(steps[1].from,steps[0].to);near(steps[2].from,steps[1].to);
        near(steps[2].to,target,i<4?3e-8:1e-10);
        for(const step of steps){
            // Each elementary rotation leaves its own printed axis unchanged.
            const col='XYZ'.indexOf(step.axis);
            for(let r=0;r<3;r++)assert.ok(Math.abs(step.from[r][col]-step.to[r][col])<1e-12);
            assert.ok(Number.isFinite(step.degrees));
        }
    }
});

test('three-axis animation pauses between Y, X, Z and does not chase live detector jitter',async()=>{
    const {PoseSequence,multiply,axisRotation,POSE_RED,POSE_GREEN}=await import('../stereo_calibration/static/handeye_direction.js');
    const y=.3,x=-.2,z=.1,ry=axisRotation('Y',y),ryx=multiply(ry,axisRotation('X',x));
    const target=multiply(ryx,axisRotation('Z',z)),seq=new PoseSequence();
    const feed=t=>{seq.update({key:'ordered',current:t?axisRotation('Z',.01):I,target,observedAt:t},t);return seq.sample(t);};
    const near=(a,b)=>a.forEach((row,i)=>row.forEach((v,j)=>assert.ok(Math.abs(v-b[i][j])<1e-12)));
    assert.deepEqual(feed(0).colour,POSE_RED);near(feed(1999).rotation,I);
    let f=feed(3000);assert.equal(f.axis,'Y');near(f.rotation,axisRotation('Y',y/2));
    f=feed(4200);assert.equal(f.phase,'pause');assert.equal(f.axis,'Y');near(f.rotation,ry);
    f=feed(5800);assert.equal(f.axis,'X');near(f.rotation,multiply(ry,axisRotation('X',x/2)));
    f=feed(7000);assert.equal(f.phase,'pause');near(f.rotation,ryx);
    f=feed(8600);assert.equal(f.axis,'Z');near(f.rotation,multiply(ryx,axisRotation('Z',z/2)));
    assert.equal(f.label,'Demo 3 / 3 · Turn');
    f=feed(9600);assert.equal(f.phase,'target');assert.deepEqual(f.colour,POSE_GREEN);near(f.rotation,target);
    f=feed(11600);assert.equal(f.phase,'current');near(f.rotation,axisRotation('Z',.01));
    // Render the exact same plan statically for reduced-motion users.
    feed(17400);const still=seq.sample(17400,true);assert.equal(still.axis,'X');near(still.rotation,still.steps[1].to);
});

test('remaining angles update on every measured pose even during the frozen red demonstration',async()=>{
    const {PoseSequence,axisRotation,chooseTiltOrder,multiply}=await import('../stereo_calibration/static/handeye_direction.js');
    const seq=new PoseSequence(),target=axisRotation('Y',.5);
    const update=(angle,t)=>seq.update({key:'live',current:axisRotation('Y',angle),target,observedAt:t},t);
    update(0,0);seq.sample(0);
    update(.2,500);assert.deepEqual(seq.sample(500).rotation,I);
    assert.ok(Math.abs(seq.live(500).steps[0].angle-.3)<1e-12);
    update(.4,1000);assert.ok(Math.abs(seq.live(1000).steps[0].angle-.1)<1e-12);
    assert.equal(seq.sample(1000).phase,'current');
    assert.equal(seq.live(4000),null);
    // Largest tilt first, then the other tilt, with turn always last.
    assert.equal(chooseTiltOrder(I,multiply(axisRotation('X',.4),axisRotation('Y',.1))),'XYZ');
    assert.equal(chooseTiltOrder(I,multiply(axisRotation('X',.1),axisRotation('Y',.4))),'YXZ');
    update(.6,4200);assert.equal(seq.sample(4200).phase,'current'); // Overshoot replans direction.
    assert.equal(seq.live(4200).steps[0].cue.direction,'away');
    const changed=multiply(axisRotation('X',.5),axisRotation('Y',.5));
    seq.update({key:'live',current:changed,target,observedAt:4500},4500);
    assert.equal(seq.order,'YXZ'); // Do not reorder as live errors change.
});

test('live edge readings replace the single total-angle line and clear when stale',async()=>{
    const {updatePosePreview,animatePosePreview,axisRotation,multiply}=await import('../stereo_calibration/static/handeye_direction.js');
    const nodes=nodesFor('handeye.html'),sideways=axisRotation('Z',-Math.PI/2);
    const state={workspace_id:'live-ui',run_id:'r',board:{columns:4,rows:5},measurement:{frame_id:1,age_s:0}};
    const g={key:'first',adjustment:{kind:'tilt'},pose_preview:{current_rotation:sideways,target_rotation:multiply(sideways,axisRotation('Y',.4))}};
    updatePosePreview(state,g,0);
    assert.equal(nodes.get('guideError').hidden,true);
    assert.equal(nodes.get('poseSteps').hidden,false);
    assert.equal(nodes.get('poseNameY').textContent,'1. Top-edge tilt');
    assert.equal(nodes.get('poseRemainingY').textContent,'22.9°');
    assert.equal(nodes.get('poseDirectionY').textContent,'toward camera');
    state.measurement.frame_id=2;g.pose_preview.current_rotation=multiply(sideways,axisRotation('Y',.2));
    updatePosePreview(state,g,500);
    assert.equal(nodes.get('poseRemainingY').textContent,'11.5°');
    assert.equal(nodes.get('posePreview').dataset.phase,'current');
    state.hold_progress=.5;updatePosePreview(state,g,600);
    assert.equal(nodes.get('poseSteps').hidden,false); // Still visible while capture checks run.
    assert.equal(nodes.get('poseRemainingY').textContent,'11.5°');
    animatePosePreview(3600);
    assert.equal(nodes.get('poseRemainingY').textContent,'—');
    assert.equal(nodes.get('poseSteps').dataset.stale,'true');
    assert.equal(nodes.get('poseReadoutStatus').textContent,'Waiting for a detected board');
    g.adjustment={kind:'move'};state.measurement.frame_id=3;
    updatePosePreview(state,g,4000);
    assert.equal(nodes.get('poseSteps').hidden,false); // Angles remain available outside the optional circle.
    assert.equal(nodes.get('poseRemainingY').textContent,'11.5°');
    assert.equal(nodes.get('posePreview').hidden,true); // Position guidance still uses its original picture.
});

test('single-axis depth cues remain attached to the correct printed edge on a sideways board',async()=>{
    const {axisPlan,multiply,axisRotation}=await import('../stereo_calibration/static/handeye_direction.js');
    const sideways=axisRotation('Z',-Math.PI/2);
    for(const axis of ['Y','X'])for(const sign of [1,-1]){
        const target=multiply(sideways,axisRotation(axis,sign*.2));
        const s=axisPlan(sideways,target).find(s=>s.axis===axis);
        assert.equal(s.cue.edge,axis==='Y'?'right':'bottom');
        assert.equal(s.cue.direction,(axis==='Y'?sign>0:sign<0)?'toward':'away');
    }
});


test('guided calibration progress is governed by coverage and prediction, not pose counts',()=>{
    const nodes=nodesFor('handeye.html');
    const state={stage:'training',training_count:30,check_count:0,
        readiness:{ready:false,reason:'Spread rotations across another axis.',
            coverage:{rotation_span_deg:47,sufficient:false},fit_max_rms_px:1.2,
            leave_one_out_max_rms_px:7.2,threshold_px:3}};
    renderWorkflow(state);
    assert.equal(nodes.get('stageLabel').textContent,'GUIDED CALIBRATION');
    assert.equal(nodes.get('count').textContent,'30');
    assert.equal(nodes.get('countLabel').textContent,'30 for fitting · 0 fresh validation');
    assert.equal(nodes.get('coverageStatus').textContent,'47° span · vary axes');
    assert.equal(nodes.get('readinessReason').textContent,state.readiness.reason);
    assert.equal(nodes.get('fitError').textContent,'1.20 px');
    assert.equal(nodes.get('validationError').textContent,'7.20 px');
    assert.equal(nodes.get('validationError').classes.get('bad'),true);
    assert.doesNotMatch(nodes.get('instructions').textContent,/at least|12|five|Next|Solve/);
});

test('automatic fit transition keeps one guided workflow and explains fresh validation',()=>{
    const nodes=nodesFor('handeye.html');
    const state={stage:'training',training_count:9,check_count:0,
        readiness:{ready:true,reason:'Coverage and withheld predictions qualify.',
            coverage:{rotation_span_deg:54,sufficient:true},fit_max_rms_px:1.5,
            leave_one_out_max_rms_px:2.2,threshold_px:3}};
    renderWorkflow(state);
    assert.equal(nodes.get('stageLabel').textContent,'GUIDED CALIBRATION');
    state.stage='checking';state.check_count=1;
    state.report={reason:'Add a different rotation axis for validation.',
        coverage:{rotation_span_deg:16,sufficient:false},max_rms_px:1.8,
        checks:[{passed:true,left_rms_px:1.8,right_rms_px:1.2}]};
    renderWorkflow(state);
    assert.equal(nodes.get('stageLabel').textContent,'GUIDED CALIBRATION');
    assert.equal(nodes.get('title').textContent,'Continue with new poses');
    assert.match(nodes.get('instructions').textContent,/candidate is now fixed/);
    assert.match(nodes.get('instructions').textContent,/without being used to fit/);
    assert.equal(nodes.get('validationLabel').textContent,'Fresh-pose error');
    assert.equal(nodes.get('validationError').textContent,'1.80 px');
    assert.equal(nodes.get('readinessReason').textContent,state.report.reason);
    assert.equal(nodes.get('count').textContent,'10');
    assert.equal(nodes.has('solve'),false);
});

test('provisional fit retains its error warning and requests more tilt coverage',()=>{
    const nodes=nodesFor('handeye.html');
    renderWorkflow({stage:'training',training_count:6,check_count:0,
        readiness:{ready:false,status:'stability',cross_validation_coverage_sufficient:false,
            reason:'Add different wrist tilts; this fit is provisional.',
            coverage:{rotation_span_deg:75,sufficient:true},fit_max_rms_px:3.094,
            leave_one_out_max_rms_px:null,threshold_px:3}});
    assert.equal(nodes.get('fitError').textContent,'3.09 px · provisional');
    assert.equal(nodes.get('fitError').classes.get('bad'),true);
    assert.equal(nodes.get('validationError').textContent,'Needs more tilts');
    assert.match(nodes.get('readinessReason').textContent,/provisional/);
    assert.equal(nodes.get('title').textContent,'Move, pause, calibrate');
});

test('restored accepted observations never present the archived attempt as a passed calibration',()=>{
    const nodes=nodesFor('handeye.html');
    const state={stage:'training',training_count:12,check_count:0,
        recovery:{parent_run_id:'previous-attempt',checkpoint_sample_id:'0011_abc',archived_check_count:4},
        readiness:{ready:false,reason:'Fit error exceeds the image budget.',
            fit_max_rms_px:8.3,leave_one_out_max_rms_px:null,threshold_px:3}};
    renderWorkflow(state);
    assert.equal(nodes.get('recoveryNotice').hidden,false);
    assert.match(nodes.get('recoveryNotice').textContent,/4 failed checks remain in archived attempt previous-attempt/);
    assert.match(nodes.get('recoveryNotice').textContent,/archived attempt did not pass calibration/);
    assert.equal(nodes.get('fitError').classes.get('bad'),true);
    assert.equal(nodes.get('validationError').textContent,'—');
    assert.equal(nodes.get('readinessReason').textContent,state.readiness.reason);
    state.recovery=null;renderWorkflow(state);
    assert.equal(nodes.get('recoveryNotice').hidden,true);
});

test('only completed geometric validation is shown as passed, without physical accuracy claims',()=>{
    const nodes=nodesFor('handeye.html');
    renderWorkflow({stage:'complete',training_count:10,check_count:6,
        readiness:{fit_max_rms_px:1.1,threshold_px:3},
        report:{max_rms_px:2.2,coverage:{rotation_span_deg:43,sufficient:true},checks:[]}});
    assert.equal(nodes.get('stageLabel').textContent,'RESULT');
    assert.equal(nodes.get('title').textContent,'Geometric checks passed');
    assert.equal(nodes.get('readinessReason').textContent,'Coverage and held-out image checks passed.');
    assert.match(nodes.get('instructions').textContent,/Nothing is applied to the robot automatically/);
});

test('completed calibration does not ask for more movement or board detection',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'done',run_id:'r',stage:'complete',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,capture_ready:false,
        measurement:{frame_id:1,age_s:0,reason:'Pose already captured. Change position or wrist tilt, then pause.'},
        guidance:{key:'done',title:'Calibration verified',instruction:'Collection is complete.',
            axis_span_deg:[0,0,0],adjustment:{kind:'hold'}}};
    renderGuide(state,false,1000);renderGuide(state,false,2200);
    assert.equal(nodes.get('message').textContent,'Calibration is complete. No additional poses are needed.');
    assert.equal(nodes.get('guideError').textContent,'');
    assert.equal(nodes.get('poseLegend').hidden,true);
    assert.equal(nodes.get('badge').textContent,'Passed');
});


test('background assessment is explicit while retaining measured coverage and pixel results',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'assessment',run_id:'r',stage:'training',assessing:true,
        feedback:{connected:true},board:{columns:4,rows:5,square_mm:25},auto_capture:true,
        capture_ready:false,training_count:8,check_count:0,just_captured:true,
        measurement:{reason:'Ready to capture this steady pose.',frame_id:1,age_s:0},
        readiness:{fit_max_rms_px:2.4,leave_one_out_max_rms_px:3.2,threshold_px:3,
            coverage:{rotation_span_deg:37,sufficient:true,translation_span_m:.12}},
        guidance:{key:'manual',title:'Choose another pose',instruction:'Show board',
            axis_span_deg:[20,30,10],adjustment:{kind:'hold'}}};
    renderWorkflow(state);renderGuide(state,false,1000);
    assert.match(nodes.get('readinessReason').textContent,/Assessing rotation coverage/);
    assert.equal(nodes.get('fitError').textContent,'2.40 px');
    assert.equal(nodes.get('coveragePosition').textContent,'120 mm');
    assert.equal(nodes.get('badge').textContent,'Assessing');
    assert.equal(nodes.get('capture').disabled,true);
    assert.match(nodes.get('message').textContent,/being assessed.*You can move/);
});


test('fresh validation coverage never inherits sufficient training coverage',()=>{
    const nodes=nodesFor('handeye.html');
    renderWorkflow({stage:'checking',training_count:10,check_count:0,
        readiness:{ready:true,coverage:{rotation_span_deg:56,sufficient:true},
            fit_max_rms_px:1.1,leave_one_out_max_rms_px:1.4},
        diversity:{rotation_span_deg:0,sufficient:false},report:null});
    assert.equal(nodes.get('coverageStatus').textContent,'0° span · vary axes');
    assert.equal(nodes.get('validationError').textContent,'—');
    assert.equal(nodes.get('fitError').textContent,'1.10 px');
});


test('coverage metrics use the backend observability policy, not printed-axis bar quotas',()=>{
    const nodes=nodesFor('handeye.html');
    const state={stage:'training',training_count:20,check_count:0,
        policy:{minimum_rotation_span_deg:25,second_rotation_singular_value_rad:.15,check_reprojection_rms_px:3},
        readiness:{ready:false,reason:'Add another independent tilt axis.',threshold_px:3,
            coverage:{rotation_span_deg:80,second_axis_rms_deg:0,sufficient:false,translation_span_m:.075}}};
    renderWorkflow(state);
    assert.equal(nodes.get('coverageSpan').textContent,'80.0°');
    assert.equal(nodes.get('coverageSpanTarget').textContent,'25.0° minimum');
    assert.equal(nodes.get('coverageExcitation').textContent,'0.0° RMS');
    assert.equal(nodes.get('coverageExcitationTarget').textContent,'8.6° RMS minimum');
    assert.equal(nodes.get('coveragePosition').textContent,'75 mm');
    assert.match(nodes.get('coverageStatus').textContent,/vary axes/);
    assert.match(nodes.get('pixelBudget').textContent,/3.0 px operational budget/);
    assert.match(nodes.get('pixelBudget').textContent,/not a universal standard or a physical accuracy/);
    // A new deployment's thresholds must be visible without changing JavaScript.
    state.policy.minimum_rotation_span_deg=30;state.policy.second_rotation_singular_value_rad=.2;
    renderWorkflow(state);
    assert.equal(nodes.get('coverageSpanTarget').textContent,'30.0° minimum');
    assert.equal(nodes.get('coverageExcitationTarget').textContent,'11.5° RMS minimum');
    assert.equal(nodes.has('coverageXFill'),false);
});

test('review and unavailable guidance retain a labelled image reference without a motion target',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'no-suggestion',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,capture_ready:false,
        measurement:{reason:'Board is visible.',frame_id:1,age_s:0},
        guidance:{key:'review',title:'Review inconsistent measurements',instruction:'Inspect the saved pose pairing.',
            axis_span_deg:[40,40,0],adjustment:{kind:'tilt',target:[.5,.5]},
            position_target:[.5,.5],position_radius:.14,
            pose_preview:{current_rotation:I,target_rotation:I}}};
    renderWorkflow(state);renderGuide(state,false,0);
    for(const id of ['guidePicture','posePreview','guideNext'])assert.equal(nodes.get(id).hidden,true,id);
    assert.equal(nodes.get('livePositionTarget').attributes.visibility,'hidden');
    assert.equal(nodes.get('positionOverlay').hidden,false);
    assert.equal(nodes.get('positionLegend').hidden,false);
    assert.equal(nodes.get('positionOverlay').dataset.mode,'reference');
    assert.equal(nodes.get('positionTargetDot').style.left,'25%');
    assert.equal(nodes.get('positionTargetDot').style.top,'50%');
    assert.equal(nodes.get('positionLegendText').textContent,'Blue dot: image centre · guidance paused');
    assert.equal(nodes.get('guideTitle').textContent,'Pose guide paused');
    assert.equal(nodes.get('guidePaused').hidden,false);
    assert.equal(nodes.get('reviewDetails').hidden,false);
    assert.match(nodes.get('guideInstruction').textContent,/Check the board mount/);
    assert.equal(nodes.get('guideReason').textContent,state.guidance.instruction);
    assert.equal(nodes.get('title').textContent,'Review image agreement');
    assert.equal(nodes.get('autoCapture').disabled,true);
    assert.equal(nodes.get('autoCapture').textContent,'Review required');
    assert.doesNotMatch(nodes.get('instructions').textContent,/Start guided capture/);
    state.guidance.key='unavailable';state.guidance.title='No suitable suggestion from here';
    state.guidance.instruction='Choose a comfortable visible pose manually.';
    renderGuide(state,false,500);
    assert.equal(nodes.get('guideNext').hidden,false);
    assert.equal(nodes.get('guidePicture').hidden,true);
    assert.equal(nodes.get('positionOverlay').hidden,false);
    assert.equal(nodes.get('livePositionTarget').attributes.visibility,'hidden');
    assert.equal(nodes.get('positionLegendText').textContent,'Blue dot: image centre · waiting for a position guide');
    assert.equal(nodes.get('poseSteps').hidden,false);
    assert.equal(nodes.get('poseRemainingY').textContent,'—');
    assert.equal(nodes.get('guideTitle').textContent,state.guidance.title);
    assert.equal(nodes.get('guideInstruction').textContent,state.guidance.instruction);
    state.guidance.key='local-2';state.guidance.position_target=[.6,.4];
    state.guidance.adjustment={kind:'tilt',edge:'right'};
    renderGuide(state,false,1000);
    assert.equal(nodes.get('positionOverlay').dataset.mode,'target');
    assert.equal(nodes.get('livePositionTarget').attributes.visibility,'visible');
    assert.equal(nodes.get('positionTargetDot').style.left,'30%');
    assert.equal(nodes.get('positionTargetDot').style.top,'40%');
    assert.equal(nodes.get('guideTitle').textContent,'Tilt the board');
    assert.doesNotMatch(nodes.get('guideInstruction').textContent,/optional/i);
});

test('archived failed views remain visible as regression evidence without changing active counts',()=>{
    const nodes=nodesFor('handeye.html');
    const state={stage:'training',training_count:12,check_count:0,
        readiness:{threshold_px:3,ready:false,reason:'Previous views still fail.',
            prior_validation:{passed:false,max_rms_px:7.4,reason:'Check pose 0012 still disagrees.'}}};
    renderWorkflow(state);
    assert.equal(nodes.get('priorValidationNotice').hidden,false);
    assert.match(nodes.get('priorValidationNotice').textContent,/Previous failed views still disagree/);
    assert.match(nodes.get('priorValidationNotice').textContent,/7.40 px/);
    assert.equal(nodes.get('count').textContent,'12');
    assert.equal(nodes.get('countLabel').textContent,'12 for fitting · 0 fresh validation');
    state.readiness.prior_validation.passed=true;
    state.readiness.prior_validation.max_rms_px=2.1;
    renderWorkflow(state);
    assert.match(nodes.get('priorValidationNotice').textContent,/never used for fitting/);
});

test('negligible rotation components do not become full animation steps',async()=>{
    const {PoseSequence,axisRotation,multiply,updatePosePreview}=await import('../stereo_calibration/static/handeye_direction.js');
    const target=multiply(multiply(axisRotation('Y',20*Math.PI/180),axisRotation('X',.01*Math.PI/180)),axisRotation('Z',.01*Math.PI/180));
    const seq=new PoseSequence();
    const feed=t=>{seq.update({key:'deadband',current:I,target,observedAt:t},t);return seq.sample(t);};
    feed(0);
    assert.equal(feed(3000).axis,'Y');
    assert.equal(feed(4000).phase,'target'); // No2second X/Z demos for 0.01°.
    const nodes=nodesFor('handeye.html');
    updatePosePreview({workspace_id:'deadband-ui',run_id:'r',board:{columns:4,rows:5},measurement:{frame_id:1,age_s:0}},
        {key:'deadband',adjustment:{kind:'tilt'},pose_preview:{current_rotation:I,target_rotation:target}},0);
    assert.equal(nodes.get('poseDirectionX').textContent,'Within 1°');
    assert.equal(nodes.get('poseDirectionZ').textContent,'Within 1°');
});

test('material movement replans the measured snapshot while brief pose noise does not',async()=>{
    const {PoseSequence,axisRotation}=await import('../stereo_calibration/static/handeye_direction.js');
    const target=axisRotation('Y',20*Math.PI/180),seq=new PoseSequence();
    const feed=(degrees,time)=>{const current=axisRotation('Y',degrees*Math.PI/180);seq.update({key:'material',current,target,observedAt:time},time);return seq.sample(time);};
    feed(0,0);
    feed(-10,500); // Large one-frame detection jump does not instantly restart.
    assert.deepEqual(feed(-2,1000).rotation,I); // Return below4° releases the pending reset.
    feed(-10,1500);
    const replanned=feed(-7,2000); // Persists beyond200ms, including8°→4° hysteresis band.
    assert.equal(replanned.phase,'current');
    assert.match(replanned.label,/Snapshot/);
    const expected=axisRotation('Y',-7*Math.PI/180);
    expected.forEach((row,i)=>row.forEach((value,j)=>assert.ok(Math.abs(replanned.rotation[i][j]-value)<1e-12)));
    assert.deepEqual(feed(-6.5,2500).rotation,replanned.rotation); // Sub-degree jitter leaves the demo steady.
    const changedTarget=axisRotation('Y',30*Math.PI/180);
    seq.update({key:'material',current:expected,target:changedTarget,observedAt:2700},2700);
    assert.equal(seq.sample(2700).phase,'current');
    assert.ok(Math.abs(seq.live(2700).steps[0].degrees-37)<1e-10);
});


test('accuracy review takes priority over start and just-captured progress invitations',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'review-cue',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,capture_ready:true,
        readiness:{ready:false,reason:'Prediction error exceeds the budget.'},
        measurement:{reason:'Ready to capture this steady pose.',frame_id:1,age_s:0},
        guidance:{key:'review',title:'Review prediction accuracy',instruction:'Inspect the image pairing.',
            adjustment:{kind:'unavailable'},axis_span_deg:[0,0,0]}};
    renderWorkflow(state);renderGuide(state,false,0);renderGuide(state,false,1000);
    assert.equal(nodes.get('holdLabel').textContent,'Recording paused · review prediction accuracy');
    assert.equal(nodes.get('autoCapture').disabled,true);
    assert.equal(nodes.get('badge').textContent,'Needs review');
    assert.equal(nodes.get('coverageMotionHint').hidden,true);
    assert.equal(nodes.get('message').textContent,state.measurement.reason); // Current view stays distinguishable.
    state.just_captured=true;state.hold_progress=1;state.message='Captured. You can move to the next pose.';
    renderGuide(state,false,1500);
    assert.equal(nodes.get('holdLabel').textContent,'Recording paused · review prediction accuracy');
    assert.equal(nodes.get('holdFill').style.width,'0%');
    assert.equal(nodes.get('holdArea').classes.get('captured'),false);
    assert.equal(nodes.get('capture').textContent,'Capture this pose manually');
    assert.equal(nodes.get('badge').textContent,'Needs review');
    assert.doesNotMatch(nodes.get('message').textContent,/move|next pose/i);
    assert.match(nodes.get('message').textContent,/Review the prediction error/);
});

test('rejected attempts retain a retry action instead of flashing captured or disappearing',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'retry-ui',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,capture_ready:true,
        just_captured:true,hold_progress:1,message:'Captured. Move to the next pose.',
        can_retry_capture:true,can_exclude_failed_training:true,failed_training_ids:['bad'],
        capture_issue:{kind:'training_accuracy',status:'retry_required',message:'Image prediction failed. Check the board and retry.'},
        measurement:{reason:'Ready to capture this steady pose.',frame_id:1,age_s:0},
        guidance:{key:'review',title:'Review prediction accuracy',instruction:'Review the measurements.',adjustment:{kind:'unavailable'}}};
    renderGuide(state,false,0);renderGuide(state,false,1500);
    assert.equal(nodes.get('retryCapture').hidden,false);
    assert.equal(nodes.get('retryCapture').disabled,false);
    assert.equal(nodes.get('guideInstruction').textContent,state.capture_issue.message);
    assert.equal(nodes.get('badge').textContent,'Retry needed');
    assert.equal(nodes.get('holdArea').classes.get('captured'),false);
    assert.equal(nodes.get('holdFill').style.width,'0%');
    assert.equal(nodes.get('autoCapture').disabled,true);
    state.just_captured=false;state.measurement.reason='Show more of the board.';
    renderGuide(state,false,4500);
    assert.equal(nodes.get('guideInstruction').textContent,state.capture_issue.message);
    assert.equal(nodes.get('retryCapture').hidden,false);
    state.pending=true;state.can_retry_capture=false;state.capture_issue.status='retry_pending';
    state.capture_issue.message='Waiting for clear, steady measurements.';
    renderGuide(state,false,5000);
    assert.equal(nodes.get('retryCapture').disabled,true);
    assert.equal(nodes.get('badge').textContent,'Retrying…');
    state.pending=false;state.capture_issue.status='resolved';state.guidance.key='reference';
    renderGuide(state,false,6000);
    assert.equal(nodes.get('retryCapture').hidden,true);
    assert.equal(nodes.get('autoCapture').disabled,false);
});

test('automatic retry stays visible, retains the dwell and offers stop instead of a retry click',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'auto-retry-ui',run_id:'r',stage:'training',feedback:{connected:false},
        board:{columns:4,rows:5,square_mm:25},auto_capture:true,capture_ready:false,
        just_captured:true,hold_progress:0,can_retry_capture:true,
        capture_issue:{kind:'capture_timeout',status:'retry_waiting',retryable:true,
            message:'The previous attempt timed out. Automatic retry will wait for valid measurements.'},
        measurement:{reason:'Waiting for measured arm feedback.',frame_id:1,age_s:0},
        guidance:{key:'unavailable',title:'Show the board',instruction:'Show the complete board.',adjustment:{kind:'unavailable'}}};
    renderGuide(state,false,0);
    assert.equal(nodes.get('retryCapture').hidden,true);
    assert.equal(nodes.get('guideTitle').textContent,'Will retry automatically');
    assert.equal(nodes.get('badge').textContent,'Waiting to retry');
    assert.equal(nodes.get('guideInstruction').textContent,state.capture_issue.message);
    assert.equal(nodes.get('guidePausedLabel').textContent,'Waiting');
    assert.equal(nodes.get('guideNext').hidden,true);
    assert.equal(nodes.get('autoCapture').hidden,false);
    assert.equal(nodes.get('autoCapture').disabled,false);
    assert.equal(nodes.get('autoCapture').textContent,'Stop automatic capture');
    assert.equal(nodes.get('holdLabel').textContent,'Waiting for measured arm feedback');
    assert.equal(nodes.get('holdArea').classes.get('captured'),false);
    assert.equal(nodes.get('holdFill').style.width,'0%');
    state.feedback.connected=true;state.capture_ready=true;state.hold_progress=.5;
    renderGuide(state,false,1500);
    assert.equal(nodes.get('guideTitle').textContent,'Hold still — retrying');
    assert.equal(nodes.get('holdFill').style.width,'50%');
    assert.equal(nodes.get('holdPercent').textContent,'50%');
    assert.equal(nodes.get('retryCapture').hidden,true);
    state.auto_capture=false;state.hold_progress=0;state.capture_issue.status='retry_required';
    renderGuide(state,false,2000);
    assert.equal(nodes.get('guideTitle').textContent,'Automatic retry paused');
    assert.equal(nodes.get('autoCapture').textContent,'Resume automatic capture');
    assert.equal(nodes.get('autoCapture').disabled,false);
    assert.equal(nodes.get('retryCapture').hidden,true);
    assert.equal(nodes.get('badge').textContent,'Paused');
    state.capture_issue.status='resolved';state.guidance.key='reference';
    renderGuide(state,false,3000);
    assert.equal(nodes.get('autoCapture').textContent,'Start guided capture');
});

test('retry timeout and pending attempt cannot conceal why calibration animation is paused',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'review-retry',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,capture_ready:true,
        can_retry_capture:true,just_captured:false,
        recording_review:{status:'accuracy',fit_max_rms_px:27.37,threshold_px:3,
            reason:'Saved measurements disagree.'},
        capture_issue:{kind:'capture_timeout',status:'retry_required',message:'No pose saved: stale camera.'},
        measurement:{reason:'Ready to capture this steady pose.',frame_id:1,age_s:0},
        guidance:{key:'review',title:'Review prediction accuracy',instruction:'Review measurements.',adjustment:{kind:'unavailable'}}};
    renderGuide(state,false,0);
    assert.equal(nodes.get('guideTitle').textContent,'Saved poses do not agree');
    assert.match(nodes.get('guideInstruction').textContent,/27\.4 px.*3\.0 px/);
    assert.doesNotMatch(nodes.get('guideInstruction').textContent,/stale/);
    assert.equal(nodes.get('captureAttemptStatus').textContent,'Last retry timed out; no pose was saved.');
    assert.equal(nodes.get('guidePausedHint').textContent,'Movement guide paused');
    assert.equal(nodes.get('badge').textContent,'Needs review');
    assert.equal(nodes.get('retryCapture').textContent,'Capture one diagnostic pose');
    assert.equal(nodes.get('retryCapture').disabled,false);
    state.measurement.reason='Pose already captured. Change position or wrist tilt, then pause.';
    renderGuide(state,false,500);
    assert.equal(nodes.get('message').textContent,'This pose is already saved. Review the calibration issue above before continuing.');
    state.pending=true;state.can_retry_capture=false;
    state.capture_issue.status='retry_pending';state.capture_wait_reason='Wrist is moving. Hold still.';
    renderGuide(state,false,1000);
    assert.equal(nodes.get('guideTitle').textContent,'Saved poses do not agree');
    assert.equal(nodes.get('captureAttemptStatus').textContent,'Diagnostic capture waiting: Wrist is moving. Hold still.');
    assert.equal(nodes.get('retryCapture').disabled,true);
    state.pending=false;state.just_captured=true;
    state.capture_issue={kind:'training_accuracy',status:'retry_required',message:'Fit still failed.'};
    renderGuide(state,false,1500);
    assert.equal(nodes.get('captureAttemptStatus').textContent,'Pose saved. Calibration was rechecked and still needs review.');
    state.pending=false;state.recording_review=null;state.guidance.key='reference';
    state.capture_issue.status='resolved';
    renderGuide(state,false,2000);
    assert.equal(nodes.get('captureAttemptStatus').hidden,true);
});

test('camera-only placement remains visible during review without implying recording or calibration success',()=>{
    const nodes=nodesFor('handeye.html');
    const a=12*Math.PI/180, R=[[Math.cos(a),0,Math.sin(a)],[0,1,0],[-Math.sin(a),0,Math.cos(a)]];
    const state={workspace_id:'placement-review',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,capture_ready:false,
        can_retry_capture:true,just_captured:false,
        recording_review:{status:'accuracy',fit_max_rms_px:41.24,threshold_px:3,reason:'Saved poses disagree.'},
        capture_issue:{kind:'training_accuracy',status:'retry_required',message:'Fit failed.'},
        measurement:{reason:'Pose already captured. Change position or wrist tilt, then pause.',frame_id:1,age_s:0},
        guidance:{key:'review',title:'Review',instruction:'Review measurements.',adjustment:{kind:'unavailable'}},
        placement_guidance:{key:'placement-1',title:'Tilt',instruction:'Small visible tilt.',at_target:false,
            adjustment:{kind:'tilt',edge:'right'},position_target:[.75,.6],position_radius:.14,
            pose_preview:{current_rotation:I,target_rotation:R}}};
    renderGuide(state,false,0);
    assert.equal(nodes.get('guidePicture').hidden,false);
    assert.equal(nodes.get('guidePaused').hidden,true);
    assert.equal(nodes.get('posePreview').hidden,false);
    assert.equal(nodes.get('poseSteps').hidden,false);
    assert.equal(nodes.get('positionOverlay').hidden,false);
    assert.equal(nodes.get('positionOverlay').dataset.mode,'target');
    assert.equal(nodes.get('positionTargetDot').style.left,'37.5%');
    assert.equal(nodes.get('guideTitle').textContent,'Follow the small tilt below');
    assert.equal(nodes.get('recordingNotice').hidden,false);
    assert.match(nodes.get('recordingNoticeText').textContent,/41\.2 px.*3\.0 px/);
    assert.equal(nodes.get('badge').textContent,'Needs review');
    assert.equal(nodes.get('autoCapture').hidden,true);
    assert.equal(nodes.get('guideNext').hidden,false);
    assert.equal(nodes.get('holdArea').hidden,true);
    assert.match(nodes.get('message').textContent,/automatic recording remains paused/);
    state.placement_guidance.at_target=true;
    renderGuide(state,false,1000);
    assert.equal(nodes.get('guideTitle').textContent,'Position matched');
    assert.equal(nodes.get('posePhase').textContent,'Position matched · recording paused');
    assert.doesNotMatch(nodes.get('poseDepthCue').textContent,/capture checks/);
    state.placement_guidance={key:'unavailable',instruction:'Show the full board in both cameras.',adjustment:{kind:'unavailable'},visibility_counts:{left:6,right:12,shared:6,expected:12}};
    state.measurement.age_s=4;
    renderGuide(state,false,4000);
    assert.equal(nodes.get('positionOverlay').hidden,false);
    assert.equal(nodes.get('positionOverlay').dataset.mode,'reference');
    assert.equal(nodes.get('positionTargetDot').style.left,'37.5%');
    assert.match(nodes.get('positionLegendText').textContent,/reference retained/);
    assert.equal(nodes.get('posePreview').hidden,true);
    assert.equal(nodes.get('guideTitle').textContent,'Show the board clearly');
    assert.equal(nodes.get('guideNext').hidden,true);
    assert.equal(nodes.get('guideInstruction').textContent,'Show the full board in both cameras.');
    assert.equal(nodes.get('recordingNotice').hidden,false);
    assert.equal(nodes.get('leftEyeLabel').textContent,'LEFT CAMERA · SHOW FULL BOARD (6/12)');
    assert.equal(nodes.get('rightEyeLabel').textContent,'RIGHT CAMERA');
});

test('partial tracking establishes a fixed position dot and gaps never make it chase stale geometry',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'partial-position',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,capture_ready:false,
        recording_review:{reason:'Review saved poses.'},
        guidance:{key:'review',policy:{position_radius:.14},adjustment:{kind:'unavailable'}},
        placement_guidance:{key:'unavailable',instruction:'Show more corners.',adjustment:{kind:'unavailable'}},
        measurement:{frame_id:1,age_s:.2,signature:{x:.77,y:.49},reason:'Need 12 corners.'}};
    renderGuide(state,false,0);
    assert.equal(nodes.get('positionOverlay').hidden,false);
    assert.equal(nodes.get('positionTargetLabel').textContent,'Board centre here');
    assert.equal(nodes.get('positionTargetDot').style.left,'38.5%');
    assert.equal(nodes.get('positionTargetDot').style.top,'49%');
    assert.equal(nodes.get('livePositionTarget').attributes.visibility,'visible');
    state.measurement={frame_id:2,age_s:4,signature:{x:.1,y:.1},reason:'Tracking lost.'};
    renderGuide(state,false,4000);
    assert.equal(nodes.get('positionTargetDot').style.left,'38.5%');
    assert.equal(nodes.get('posePreview').hidden,true);
    state.measurement={frame_id:3,age_s:.1,signature:{x:.66,y:.4},reason:'Partial tracking.'};
    renderGuide(state,false,5000);
    assert.equal(nodes.get('positionTargetDot').style.left,'38.5%'); // Goal does not chase current centre.
    state.run_id='new';state.measurement=null;
    renderGuide(state,false,6000);
    assert.equal(nodes.get('positionTargetLabel').textContent,'Image centre');
    assert.equal(nodes.get('livePositionTarget').attributes.visibility,'hidden');
});

test('information-selected depth move shows distance and reason instead of an already-matched tilt',()=>{
    const nodes=nodesFor('handeye.html');
    const state={workspace_id:'depth-information',run_id:'r',stage:'training',feedback:{connected:true},
        board:{columns:4,rows:5,square_mm:25},auto_capture:false,capture_ready:false,
        hold_progress:0,just_captured:false,measurement:{frame_id:2,age_s:0,reason:'Wrist is moving.'},
        guidance:{key:'local-depth',title:'Move the board farther',instruction:'Move the whole board about 25 mm farther.',
            selection_reason:'Adds a different viewing distance after establishing two-axis tilt coverage.',
            translation_target_camera_m:[0,0,.525],translation_error_camera_m:[0,0,.025],
            position_target:[.5,.5],position_radius:.025,at_target:false,
            current_rotation:I,target_rotation:I,axis_span_deg:[25,25,0],
            pose_preview:{current_rotation:I,target_rotation:I},adjustment:{kind:'distance',direction:'farther'}}};
    renderGuide(state,false,0);
    assert.equal(nodes.get('guideTitle').textContent,'Move the board farther');
    assert.equal(nodes.get('guideInstruction').textContent,state.guidance.instruction);
    assert.equal(nodes.get('guideBenefit').textContent,state.guidance.selection_reason);
    assert.equal(nodes.get('guideBenefit').hidden,false);
    assert.equal(nodes.get('posePreview').hidden,true);
    assert.equal(nodes.get('poseSteps').hidden,true);
    assert.equal(nodes.get('guideGraphic').style.display,'');
});
