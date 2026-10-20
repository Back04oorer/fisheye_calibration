// One camera-view board: current (2 s), larger tilt, other tilt, turn, target.
// Illustration only. Neither time nor colour is a capture/robot command.
// Sequential pose guidance: Calibration Venus, section 3.4 (arXiv:2009.05983).
// Intrinsic board axes; the order is fixed per suggestion, readings stay live.
const $ = id => document.getElementById(id);
const text = (id,value) => {if($(id).textContent!==value)$(id).textContent=value;};
export const POSE_TIMING={current:2000,motion:2000,pause:800,target:2000,stale:3000};
// Illustration tolerances only; neither value participates in capture acceptance.
export const POSE_DEMO_TOLERANCE_DEG=1;
const SNAPSHOT_REPLAN={enterDeg:8,releaseDeg:4,settleMs:200};
export const POSE_RED=[193,70,62], POSE_GREEN=[29,135,100];
const valid = r => Array.isArray(r)&&r.length===3&&r.every(row=>Array.isArray(row)&&row.length===3&&row.every(Number.isFinite));

export function quaternion(r){
    const t=r[0][0]+r[1][1]+r[2][2];let q;
    if(t>0){const s=Math.sqrt(t+1)*2;q=[s/4,(r[2][1]-r[1][2])/s,(r[0][2]-r[2][0])/s,(r[1][0]-r[0][1])/s];}
    else {
        const i=r[0][0]>r[1][1]?(r[0][0]>r[2][2]?0:2):(r[1][1]>r[2][2]?1:2),j=(i+1)%3,k=(i+2)%3;
        const s=Math.sqrt(1+r[i][i]-r[j][j]-r[k][k])*2;
        q=[(r[k][j]-r[j][k])/s,0,0,0];q[i+1]=s/4;q[j+1]=(r[j][i]+r[i][j])/s;q[k+1]=(r[k][i]+r[i][k])/s;
    }
    const n=Math.hypot(...q);return q.map(v=>v/n);
}
export function slerp(a,b,t){
    let dot=a.reduce((v,x,i)=>v+x*b[i],0);
    if(dot<0){b=b.map(v=>-v);dot=-dot;}
    let q;
    if(dot>.9995)q=a.map((v,i)=>v+t*(b[i]-v));
    else {const theta=Math.acos(Math.min(1,dot)),s=Math.sin(theta);q=a.map((v,i)=>(v*Math.sin((1-t)*theta)+b[i]*Math.sin(t*theta))/s);}
    const n=Math.hypot(...q);return q.map(v=>v/n);
}
export function rotation(q){
    const [w,x,y,z]=q;
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
}
export function cameraToCss(r){
    // OpenCV x-right/y-down/z-away → CSS x-right/y-down/z-toward.
    // S R S changes coordinates without mirroring the printed board.
    const s=[1,1,-1],m=[];
    for(let col=0;col<3;col++){for(let row=0;row<3;row++)m.push(s[row]*r[row][col]*s[col]);m.push(0);}
    return `matrix3d(${[...m,0,0,0,1].join(',')})`;
}

export const multiply=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*b[k][j],0)));
const transpose=a=>a[0].map((_,i)=>a.map(row=>row[i]));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const rotationDistanceDeg=(a,b)=>2*Math.acos(Math.min(1,Math.abs(quaternion(multiply(transpose(a),b))[0])))*180/Math.PI;
export function axisRotation(axis,angle){
    const q=[Math.cos(angle/2),0,0,0];q['XYZ'.indexOf(axis)+1]=Math.sin(angle/2);
    return rotation(q);
}

export function axisPlan(current,target,ratio=.8,order='YXZ'){
    // R_current^T R_target = Ry Rx Rz or Rx Ry Rz. Each rotation is about
    // the printed board's axis AFTER the preceding step, never a robot joint.
    const d=multiply(transpose(current),target);
    if(!['YXZ','XYZ'].includes(order))throw Error('Unsupported tilt order');
    const middle=Math.asin(Math.max(-1,Math.min(1,order==='YXZ'?-d[1][2]:d[0][2])));
    let angles;
    if(Math.abs(Math.cos(middle))<1e-7){
        // At +/-90 degrees, the first axis and Z couple. Choose Z=0.
        angles=[order==='YXZ'?Math.atan2(-d[2][0],d[0][0]):Math.atan2(d[2][1],d[1][1]),middle,0];
    }else{
        angles=order==='YXZ'
            ?[Math.atan2(d[0][2],d[2][2]),middle,Math.atan2(d[1][0],d[1][1])]
            :[Math.atan2(-d[1][2],d[2][2]),middle,Math.atan2(-d[0][1],d[0][0])];
        const alternative=[wrap(angles[0]+Math.PI),wrap(Math.PI-middle),wrap(angles[2]+Math.PI)];
        const travel=a=>a.reduce((sum,v)=>sum+Math.abs(v),0);
        if(travel(alternative)<travel(angles))angles=alternative;
    }
    let from=current;
    return [...order].map((axis,i)=>{
        const angle=angles[i],to=multiply(from,axisRotation(axis,angle));
        const cue=depthCue(from,to,ratio,axis);
        const name=axis==='Z'?'Turn':`${screenEdge(from,axis)}-edge tilt`;
        const step={axis,angle,degrees:angle*180/Math.PI,from,to,cue,name};from=to;return step;
    });
}

export function chooseTiltOrder(current,target){
    // Largest tilt component of the shortest relative rotation first; turn last.
    // This is a UI heuristic, not an optimal/safe robot motion planner.
    const q=quaternion(multiply(transpose(current),target));
    return Math.abs(q[1])>Math.abs(q[2])?'XYZ':'YXZ';
}
export function screenEdge(r,axis){
    const col=axis==='Y'?0:1,x=r[0][col],y=r[1][col];
    return Math.abs(x)>Math.abs(y)?(x<0?'Left':'Right'):(y<0?'Top':'Bottom');
}

export function depthCue(current,target,ratio=.8,axis=null){
    const normalDot=current.reduce((sum,row,i)=>sum+row[2]*target[i][2],0);
    if(axis==='Z'||(!axis&&normalDot>Math.cos(5*Math.PI/180))){
        // Turning within an already tilted board plane changes edge depths too.
        // Name the in-plane turn, rather than incorrectly asking for more tilt.
        const turn=current.reduce((sum,row,i)=>sum+row[1]*target[i][0],0);
        const direction=turn*current[2][2]>0?'clockwise':'counterclockwise';
        return {edge:null,direction,label:Math.abs(turn)<1e-7?'No turn needed'
            :`Turn ${direction} in the board’s plane`};
    }
    // Physical edge displacement about the board centre, in camera coordinates.
    // Camera +z is farther away. Never infer this from the direction on screen.
    const right=(target[2][0]-current[2][0])*ratio;
    const bottom=target[2][1]-current[2][1];
    const edge=axis==='Y'?'right':axis==='X'?'bottom':Math.abs(right)>Math.abs(bottom)?'right':'bottom';
    const dz=edge==='right'?right:bottom;
    if(Math.abs(dz)<1e-7)return {edge:null,label:'Follow this single-axis tilt'};
    return {edge,direction:dz<0?'toward':'away',label:dz<0
        ?'Blue-marked edge: bring toward the camera'
        :'Blue-marked edge: push away from the camera'};
}

export class PoseSequence {
    constructor(){this.key=null;this.latest=null;this.cycle=null;this.aligned=false;this.order=null;this.replanSince=null;}
    update({key,current,target,observedAt,aligned=false,reference=false,ratio=.8,captureState=null},now){
        if(key!==this.key){this.key=key;this.latest=this.cycle=this.order=null;this.replanSince=null;}
        if(valid(current)&&valid(target)&&Number.isFinite(observedAt)){
            this.order??=chooseTiltOrder(current,target);
            this.latest={current,target,observedAt,ratio,steps:axisPlan(current,target,ratio,this.order)};
            // If the operator overshoots, do not keep demonstrating the wrong
            // direction until the end of the loop. Noise below 1 degree is ignored.
            if(this.cycle&&this.latest.steps.some((s,i)=>Math.abs(s.degrees)>POSE_DEMO_TOLERANCE_DEG
                &&Math.abs(this.cycle.steps[i].degrees)>POSE_DEMO_TOLERANCE_DEG
                &&s.angle*this.cycle.steps[i].angle<0)){this.cycle=null;this.replanSince=null;}
            if(this.cycle){
                const deviation=rotationDistanceDeg(this.cycle.current,current);
                // A changed goal invalidates the old demonstration immediately.
                // A material measured movement must persist into a fresh reading;
                // the lower release threshold prevents noise at the boundary from
                // repeatedly starting and cancelling the new snapshot.
                if(rotationDistanceDeg(this.cycle.target,target)>POSE_DEMO_TOLERANCE_DEG){this.cycle=null;this.replanSince=null;}
                else {
                    if(deviation>=SNAPSHOT_REPLAN.enterDeg&&this.replanSince===null)this.replanSince=observedAt;
                    if(deviation<SNAPSHOT_REPLAN.releaseDeg)this.replanSince=null;
                    if(this.replanSince!==null&&observedAt-this.replanSince>=SNAPSHOT_REPLAN.settleMs){this.cycle=null;this.replanSince=null;}
                }
            }
        }
        if(this.aligned&&!aligned){this.cycle=null;this.replanSince=null;}
        this.aligned=aligned;this.reference=reference;
        this.captureState=captureState;
    }
    live(now){
        if(!this.latest||now-this.latest.observedAt>=POSE_TIMING.stale)return null;
        return {steps:this.latest.steps,age:Math.max(0,now-this.latest.observedAt)};
    }
    sample(now,reducedMotion=false){
        const latest=this.latest;
        if(!latest||now-latest.observedAt>=POSE_TIMING.stale){this.cycle=null;return {phase:'waiting',label:'Waiting for a board pose'};}
        if(this.captureState){
            this.cycle=null;
            return {phase:this.captureState,label:this.captureState==='captured'?'Pose saved':'Hold still · recording',rotation:latest.current,colour:POSE_GREEN};
        }
        if(this.reference||this.aligned){
            this.cycle=null;
            return {phase:'aligned',label:this.reference?'Your current pose · hold still':'Pose aligned · hold still',rotation:latest.current,colour:POSE_GREEN};
        }
        if(!this.cycle||now-this.cycle.start>=this.cycle.duration){
            const steps=latest.steps;
            const moving=steps.filter(s=>Math.abs(s.degrees)>POSE_DEMO_TOLERANCE_DEG);
            const duration=POSE_TIMING.current+moving.length*POSE_TIMING.motion
                +Math.max(0,moving.length-1)*POSE_TIMING.pause+POSE_TIMING.target;
            this.cycle={...latest,steps,moving,duration,start:now};
        }
        const cycle=this.cycle;
        const elapsed=Math.max(0,now-this.cycle.start);
        let progress=0,phase='current',label='Snapshot · measured start pose',r=cycle.current,step=cycle.moving[0];
        if(elapsed>=POSE_TIMING.current){
            const motionTime=elapsed-POSE_TIMING.current;
            const index=Math.floor(motionTime/(POSE_TIMING.motion+POSE_TIMING.pause));
            step=cycle.moving[index];
            if(!step||elapsed>=cycle.duration-POSE_TIMING.target){
                progress=1;phase='target';label='Suggested pose · hold still';r=cycle.target;step=null;
            }else{
                const within=motionTime-index*(POSE_TIMING.motion+POSE_TIMING.pause);
                let fraction=Math.min(1,within/POSE_TIMING.motion);
                fraction=fraction*fraction*(3-2*fraction);
                // Reduced motion retains all the steps as discrete poses.
                if(reducedMotion)fraction=1;
                progress=(index+fraction)/cycle.moving.length;
                phase=within<POSE_TIMING.motion?'moving':'pause';
                label=`Demo ${index+1} / ${cycle.moving.length} · ${step.name}${phase==='pause'?' · pause':''}`;
                r=multiply(step.from,axisRotation(step.axis,step.angle*fraction));
            }
        }
        return {phase,label,progress,axis:step?.axis,steps:cycle.steps,cue:step?.cue,rotation:r,
            colour:POSE_RED.map((v,i)=>Math.round(v+(POSE_GREEN[i]-v)*progress))};
    }
}

const sequence=new PoseSequence();
let enabled=false,placementOnly=false,poseReadoutLabel='Live remaining',boardKey='',latestFrameKey='',observedAt=-Infinity;
export function updatePosePreview(state,g,now){
    placementOnly=!!state.placement_only;
    poseReadoutLabel=g.pose_quality?.partial
        ?`Live remaining · ${g.pose_quality.shared_corners}/${g.pose_quality.expected_corners} corners`
        :'Live remaining · updates with each detected pose';
    enabled=!['done','manual','review','unavailable'].includes(g.key);
    const translating=!!g.translation_target_camera_m;
    const showBoard=enabled&&!translating&&g.adjustment?.kind!=='move';
    $('posePreview').hidden=!showBoard;
    $('guideGraphic').style.display=showBoard?'none':'';
    $('posePhase').hidden=!showBoard;
    $('poseDepthCue').hidden=!showBoard;
    const finished=['done','manual'].includes(g.key)||state.stage==='complete';
    $('poseSteps').hidden=finished||translating;
    $('poseReadoutStatus').hidden=finished||translating;
    $('guideError').hidden=enabled;
    $('posePerspective').hidden=!showBoard;
    if(translating){
        sequence.key=null;sequence.latest=sequence.cycle=sequence.order=null;sequence.replanSince=null;
        return;
    }
    if(!enabled){
        sequence.key=null;sequence.latest=sequence.cycle=sequence.order=null;sequence.replanSince=null;
        $('poseSteps').dataset.stale='true';
        text('poseReadoutStatus','Waiting for a usable board pose · readings resume automatically');
        for(const [axis,name] of [['Y','1. Edge tilt'],['X','2. Edge tilt'],['Z','3. Turn']]){
            text(`poseName${axis}`,name);text(`poseRemaining${axis}`,'—');
            text(`poseDirection${axis}`,'Waiting for pose');$(`poseStep${axis}`).dataset.active='false';
        }
        return;
    }
    const key=JSON.stringify(state.board);
    if(key!==boardKey){
        boardKey=key;
        $('poseBoardImage').src=`/api/handeye/board.png?pattern_only=true&revision=${encodeURIComponent(key)}`;
        const ratio=state.board.columns/state.board.rows;
        const height=116/Math.max(1,ratio),width=height*ratio;
        $('poseBoard').style.width=`${width}px`;$('poseBoard').style.height=`${height}px`;
    }
    const p=g.pose_preview,frame=state.measurement;
    const frameKey=`${state.workspace_id}:${state.run_id}:${frame?.frame_id}`;
    if(p&&frameKey!==latestFrameKey){latestFrameKey=frameKey;observedAt=now-1000*(frame?.age_s??0);}
    sequence.update({key:`${state.workspace_id}:${state.run_id}:${g.key}`,
        current:p?.current_rotation,target:p?.target_rotation,observedAt,
        aligned:g.at_target,reference:g.key==='reference',ratio:state.board.columns/state.board.rows,
        captureState:state.just_captured?'captured':state.hold_progress>0?'recording':null},now);
    animatePosePreview(now);
}

export function animatePosePreview(now){
    if(!enabled)return;
    const reduced=typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
    const frame=sequence.sample(now,reduced);
    $('posePreview').dataset.phase=frame.phase;
    $('poseBoard').style.visibility=frame.rotation?'visible':'hidden';
    $('poseWaiting').hidden=!!frame.rotation;
    text('posePhase',placementOnly&&frame.phase==='aligned'?'Position matched · recording paused':frame.label);
    const live=sequence.live(now);
    text('poseReadoutStatus',live?live.age<1500?poseReadoutLabel:'Last reading · waiting for a fresh pose':'Waiting for a detected board');
    $('poseSteps').dataset.stale=String(!live||live.age>=1500);
    for(const [i,step] of (live?.steps??[]).entries()){
        const node=$(`poseStep${step.axis}`);
        node.style.order=i;
        node.title=`Rotation about printed board ${step.axis}; applied after preceding steps`;
        text(`poseName${step.axis}`,`${i+1}. ${step.name}`);
        text(`poseRemaining${step.axis}`,`${Math.abs(step.degrees).toFixed(1)}°`);
        text(`poseDirection${step.axis}`,Math.abs(step.degrees)<=POSE_DEMO_TOLERANCE_DEG?'Within 1°':step.axis==='Z'
            ?step.cue.direction:step.cue.direction?`${step.cue.direction} camera`:'Follow the marked edge');
        node.dataset.active=String(frame.axis===step.axis);
    }
    if(!live)for(const axis of 'XYZ'){
        text(`poseRemaining${axis}`,'—');text(`poseDirection${axis}`,'Waiting for detection');
        $(`poseStep${axis}`).dataset.active='false';
    }
    $('poseEdge').hidden=!frame.cue?.edge;
    if(frame.cue?.edge)$('poseEdge').dataset.edge=frame.cue.edge;
    text('poseDepthCue',frame.cue?.label??(placementOnly&&frame.rotation?'Positioning only · automatic recording paused':frame.phase==='captured'?'You can move to the next pose':frame.rotation?'Keep the board still for the capture checks':'Pose preview resumes when detection returns'));
    if(frame.rotation){
        $('poseBoard').style.transform=cameraToCss(frame.rotation);
        $('poseBoard').style.borderColor=`rgb(${frame.colour.join(',')})`;
        $('posePhase').style.color=`rgb(${frame.colour.join(',')})`;
    }else $('posePhase').style.color='';
}
