// Stereo's illustration and animation; this is guidance, never a robot command.
import {updateBoardGraphic, updateMovementGuide} from './movement_guide.js';
import {updatePosePreview} from './handeye_direction.js';
const $ = id => document.getElementById(id);
const text = (id, value) => {if ($(id).textContent !== value) $(id).textContent=value;};
let runKey='', targetKey='', holdingUntil=0;
let positionReference=null;
// Delay short quality-status oscillations, never the capture checks themselves.
export class SettledStatus {
    constructor(){this.key=null;this.since=0;this.good=false;this.message='Waiting for a stable measurement.';}
    update(ready,message,now,code){
        const key=ready?'ready':code||message;
        if(key!==this.key){this.key=key;this.since=now;}
        if(!ready&&this.good){this.good=false;this.message='Waiting for a stable measurement.';}
        if(now-this.since>=1000){this.good=ready;this.message=message;}
        return this.message;
    }
}
let qualityStatus = new SettledStatus();

const pixels=value=>Number.isFinite(value)?`${value.toFixed(2)} px`:'—';
const guidanceHidden=key=>['done','manual','review','unavailable'].includes(key);
const measuredCoverage=state=>['checking','complete'].includes(state.stage)
    ?(state.report?.coverage||state.diversity||{})
    :(state.readiness?.coverage||state.diversity||{});
// The fit and held-out validation use different data, but require the same
// operator action: move to a useful pose and hold. Counts are only an audit trail.
export function renderWorkflow(state){
    const checking=state.stage==='checking', complete=state.stage==='complete';
    const readiness=state.readiness||{}, coverage=measuredCoverage(state);
    const budget=readiness.threshold_px??state.policy?.check_reprojection_rms_px;
    const checks=state.report?.checks||[];
    const failed=checks.some(c=>!c.passed), review=state.guidance?.key==='review';
    text('stageLabel',complete?'RESULT':'GUIDED CALIBRATION');
    $('coverageMotionHint').hidden=review||complete;
    $('instructions').hidden=review;
    text('title',complete?'Geometric checks passed':review?'Review image agreement':checking?'Continue with new poses':'Move, pause, calibrate');
    text('instructions',complete
        ?'The camera-to-arm transform is saved with its validation report. Nothing is applied to the robot automatically.'
        :review
        ?'Review the saved images, board mounting and arm-pose pairing. More captures alone do not resolve disagreement between the measured views.'
        :checking
        ?'Keep moving and pausing as before. The candidate is now fixed; new poses test it without being used to fit it. Recording finishes when coverage and image agreement qualify.'
        :'Start guided capture, vary position and wrist tilt, then pause with the complete board visible. The system fits and checks automatically as useful coverage and image agreement improve.');
    const train=state.training_count||0, check=state.check_count||0;
    text('count',String(train+check));
    text('countLabel',`${train} for fitting · ${check} fresh validation${complete?' · complete':''}`);
    const span=coverage.rotation_span_deg;
    text('coverageStatus',Number.isFinite(span)
        ?`${span.toFixed(0)}° span · ${coverage.sufficient?'varied axes':'vary axes'}`
        :'Waiting for poses');
    const provisional=!checking&&!complete&&readiness.cross_validation_coverage_sufficient===false&&Number.isFinite(readiness.fit_max_rms_px);
    text('fitError',pixels(readiness.fit_max_rms_px)+(provisional?' · provisional':''));
    text('validationLabel',checking||complete?'Fresh-pose error':'Withheld-pose error');
    const errors=checks.flatMap(c=>[c.left_rms_px,c.right_rms_px]).filter(Number.isFinite);
    const validation=checking||complete
        ?(state.report?.max_rms_px??(errors.length?Math.max(...errors):null))
        :readiness.leave_one_out_max_rms_px;
    text('validationError',provisional?'Needs more tilts':pixels(validation));
    $('fitError').classList.toggle('bad',Number.isFinite(readiness.fit_max_rms_px)&&readiness.fit_max_rms_px>budget);
    $('validationError').classList.toggle('bad',Number.isFinite(validation)&&validation>budget);
    text('readinessReason',state.assessing?'Assessing rotation coverage and withheld-pose predictions…':complete?'Coverage and held-out image checks passed.'
        :checking?(state.report?.reason||(failed?'Fresh poses disagree with the fixed candidate. Review the result below.':'Continue across different tilts and positions to validate the fixed candidate.'))
        :(readiness.reason||'Waiting for accepted poses to assess coverage and image agreement.'));
    text('diversity','Coverage and image consistency determine readiness; the pose count is informational.');
    text('budgetShort',Number.isFinite(budget)?`Limit ${budget.toFixed(1)} px · native images`:'Native camera pixels');
    text('pixelBudget',`Geometric image agreement${Number.isFinite(budget)?` · ${budget.toFixed(1)} px operational budget`:''}. This is not a universal standard or a physical accuracy measurement.`);
    const prior=state.report?.prior_validation||readiness.prior_validation;
    $('priorValidationNotice').hidden=!prior;
    if(prior)text('priorValidationNotice',prior.passed
        ?`The fixed candidate also passes the archived failed views (${pixels(prior.max_rms_px)} worst RMS). These views were never used for fitting.`
        :`Previous failed views still disagree${Number.isFinite(prior.max_rms_px)?` (${pixels(prior.max_rms_px)} worst RMS)`:''}. These archived views must also pass validation.`);
    renderCoverage(state);
    const recovered=state.recovery;
    $('recoveryNotice').hidden=!recovered;
    if(recovered)text('recoveryNotice',`Resumed accepted poses through ${recovered.checkpoint_sample_id||'the saved checkpoint'}. ${recovered.archived_check_count||0} failed checks remain in archived attempt ${recovered.parent_run_id}. The archived attempt did not pass calibration.`);
}

function renderCoverage(state){
    const coverage=measuredCoverage(state), policy=state.policy||{};
    text('coverageSpan',Number.isFinite(coverage.rotation_span_deg)?`${coverage.rotation_span_deg.toFixed(1)}°`:'—');
    text('coverageSpanTarget',Number.isFinite(policy.minimum_rotation_span_deg)?`${policy.minimum_rotation_span_deg.toFixed(1)}° minimum`:'Policy unavailable');
    text('coverageExcitation',Number.isFinite(coverage.second_axis_rms_deg)?`${coverage.second_axis_rms_deg.toFixed(1)}° RMS`:'—');
    text('coverageExcitationTarget',Number.isFinite(policy.second_rotation_singular_value_rad)?`${(policy.second_rotation_singular_value_rad*180/Math.PI).toFixed(1)}° RMS minimum`:'Policy unavailable');
    text('coveragePosition',Number.isFinite(coverage.translation_span_m)?`${(coverage.translation_span_m*1000).toFixed(0)} mm`:'—');
}

function drawMotion(state,g,now){
    const run=`${state.workspace_id}:${state.run_id}`;
    if(run!==runKey){runKey=run;qualityStatus=new SettledStatus();positionReference=null;}
    const key=`${run}:${g.key}`;
    if(key!==targetKey){targetKey=key;holdingUntil=0;}
    const adjustment=g.adjustment||{kind:'unavailable'};
    if(state.hold_progress>0)holdingUntil=now+900;
    if(state.measurement?.blocker_code==='motion')holdingUntil=0;
    const holding=now<holdingUntil||state.just_captured||g.key==='done';
    updateBoardGraphic(state.board,'/api/handeye/board.png');
    updateMovementGuide(adjustment,state.board,{holding});
    updatePosePreview(state,g,now);
    const positionTarget=g.position_target||adjustment.target;
    const hasTarget=Array.isArray(positionTarget)&&positionTarget.length===2
        &&positionTarget.every(v=>Number.isFinite(v)&&v>=0&&v<=1)
        &&Number.isFinite(g.position_radius)&&g.position_radius>0
        &&!guidanceHidden(g.key);
    // Position is a fixed image-space reference, not a newly measured pose.
    // Keep it through corner dropouts; only the tilt needs full stereo tracking.
    if(hasTarget)positionReference={point:[...positionTarget],radius:g.position_radius};
    const measured=state.measurement, signature=measured?.signature;
    if(!positionReference&&Number.isFinite(measured?.age_s)&&measured.age_s<1.5
        &&measured.timing?.fresh_at_ingress!==false
        &&signature&&[signature.x,signature.y].every(v=>Number.isFinite(v)&&v>=0&&v<=1)){
        const radius=state.guidance?.policy?.position_radius;
        positionReference={point:[signature.x,signature.y],radius:Number.isFinite(radius)&&radius>0?radius:.14};
    }
    // Before any board position is known, the dot is explicitly only image centre.
    const reference=!hasTarget&&!['done','manual'].includes(g.key)&&state.stage!=='complete';
    $('positionOverlay').hidden=!hasTarget&&!reference;
    $('positionLegend').hidden=!hasTarget&&!reference;
    $('positionOverlay').dataset.mode=hasTarget?'target':'reference';
    $('positionOverlay').dataset.anchored=String(hasTarget||!!positionReference);
    $('livePositionTarget').setAttribute('visibility',hasTarget||positionReference?'visible':'hidden');
    text('positionLegendText',hasTarget?'Blue dot: board position':positionReference
        ?'Blue dot: keep the board centre here · position reference retained while tilt tracking recovers'
        :g.key==='review'
        ?'Blue dot: image centre · guidance paused'
        :'Blue dot: image centre · waiting for a position guide');
    text('positionTargetLabel',hasTarget||positionReference?'Board centre here':'Image centre');
    if(hasTarget||reference){
        const [x,y]=hasTarget?positionTarget:positionReference?.point||[.5,.5];
        const radius=hasTarget?g.position_radius:positionReference?.radius||0;
        const target=$('livePositionTarget');
        // SBS image: [0,100] is LEFT, [100,200] is RIGHT. This is a
        // fixed goal, not a stale detected-board overlay on a newer image.
        target.setAttribute('cx',x*100);target.setAttribute('cy',y*100);
        target.setAttribute('rx',radius*100);target.setAttribute('ry',radius*100);
        $('positionTargetDot').style.left=`${x*50}%`;
        $('positionTargetDot').style.top=`${y*100}%`;
        $('positionTargetLabel').style.left=`${x*50}%`;
        $('positionTargetLabel').style.top=`${y*100}%`;
    }
    const suppressed=['review','unavailable'].includes(g.key);
    const tilt=!suppressed&&(['tilt','roll'].includes(adjustment.kind)
        ||(g.pose_preview&&!['move','distance','ready'].includes(adjustment.kind)&&!['reference','done','manual'].includes(g.key)));
    $('guidePicture').hidden=suppressed;
    $('guidePaused').hidden=!suppressed;
    text('guidePausedLabel',state.auto_capture&&!['review','done'].includes(g.key)?'Waiting':'Ⅱ Paused');
    text('guidePausedHint',state.auto_capture&&!['review','done'].includes(g.key)?'Capture is automatic':'No motion target');
    $('reviewDetails').hidden=!suppressed;
    text('guideReason',g.instruction||'');
    const reason=state.report?.reason||state.readiness?.reason;
    $('guideReason').hidden=!g.instruction||!!(reason&&g.instruction.startsWith(reason));
    $('poseLegend').hidden=suppressed||state.stage==='complete';
    text('guideTitle',g.key==='review'?'Pose guide paused':suppressed?g.title:state.just_captured?'Pose saved — choose another':holding&&g.key!=='done'?'Good pose — hold still':tilt?'Tilt the board':g.title);
    text('guideInstruction',g.key==='review'?'Saved poses disagree with the camera images. Check the board mount and measurements before continuing.':suppressed?g.instruction:state.just_captured?'This pose is saved. Choose another comfortable position or wrist rotation.':holding&&g.key!=='done'?'Keep this pose steady while it is checked.':tilt
        ?'Follow the illustrated tilt, then hold still. Capture starts when the pose is ready.'
        :g.instruction);
    text('guideError',suppressed||g.key==='done'?'':adjustment.kind==='move'?'Bring the board centre towards the blue dot'
        :'Waiting for a detected board');
    if(suppressed)$('guideError').hidden=true;
}
export function renderGuide(state,busy,now=performance.now()){
    const original=state.guidance;if(!original)return;
    const placement=state.recording_review&&state.placement_guidance;
    const g=placement||original;
    text('guideBenefit',g.selection_reason||'');
    $('guideBenefit').hidden=!g.selection_reason||!!state.recording_review;
    const done=g.key==='done', complete=state.stage==='complete', review=original.key==='review';
    const failed=state.stage==='checking' && state.report?.checks?.some(c=>!c.passed);
    text('guideEyebrow',review?'GUIDANCE · PAUSED':'POSE GUIDE · YOU MOVE THE ARM');
    $('guideNext').hidden=done||g.key==='reference'||g.key==='manual'||g.key==='review';$('guideNext').disabled=busy||state.pending;
    drawMotion(placement?{...state,just_captured:false,hold_progress:0,placement_only:true}:state,g,now);
    renderCoverage(state);
    $('autoCapture').hidden=complete;$('autoCapture').disabled=busy||state.pending||(g.key==='review'&&!state.auto_capture);
    text('autoCapture',state.auto_capture?'Stop automatic capture':g.key==='review'?'Review required':'Start guided capture');
    $('holdArea').hidden=review||complete||(!state.auto_capture&&!state.just_captured);
    const progress=review?0:state.hold_progress||0;
    const measuredMessage=qualityStatus.update(state.capture_ready,state.measurement?.reason||state.message,now,state.measurement?.blocker_code);
    text('message',state.error||(complete?'Calibration is complete. No additional poses are needed.':state.assessing?'Saved poses are being assessed. You can move to the next pose.':review&&state.just_captured?'Last pose saved. Review the prediction error before continuing.':state.just_captured?state.message:measuredMessage));
    $('capture').disabled=busy||state.auto_capture||!state.capture_ready||!qualityStatus.good;
    if(review)text('capture','Capture this pose manually');
    text('badge',complete?'Passed':state.assessing?'Assessing':review?'Needs review':state.pending?'Capturing…':state.auto_capture?'Recording':'Paused');
    const cue=review?'Recording paused · review prediction accuracy':state.just_captured?'✓ Captured — you can move now':complete?'Calibration complete':state.assessing?'Checking coverage and image agreement':failed&&!state.auto_capture?'Check needs review — see results':!state.auto_capture?'Press Start for automatic recording':!state.feedback.connected?'Waiting for measured arm feedback':!state.capture_ready?'Capture waiting · reason shown above':'Valid new pose — hold still to record';
    text('holdLabel',cue);text('holdPercent',state.auto_capture&&progress>0?`${Math.round(progress*100)}%`:'');
    $('holdFill').style.width=`${!review&&state.just_captured?100:progress*100}%`;
    $('holdArea').classList.toggle('captured',!review&&state.just_captured);
    if(!review&&!state.assessing&&state.hold_progress>0)text('badge','Hold steady');
    if(!review&&!state.assessing&&state.just_captured)text('badge','Captured');
    $('checkNotice').hidden=!failed;
    if(failed){
        const checks=state.report.checks, count=checks.filter(c=>!c.passed).length;
        text('checkNotice',`${count} of ${checks.length} fresh poses failed calibration validation. ${state.auto_capture?'Recording is active.':'Automatic recording is stopped.'} Current capture readiness is shown above; it does not mean the calibration passed. See results below.`);
    }
    renderCaptureIssue(state,busy);
    renderReviewStatus(state,busy,placement);
    // Preserve the distinct automatic-wait, accuracy-review and manual-retry
    // titles above. A generic "Retry capture" title hid the actual state.
}

// A failed model and a failed recording attempt are different problems. Keep
// the model failure primary so Retry/timeouts cannot hide why motion is paused.
function renderReviewStatus(state,busy,placement){
    const review=state.recording_review||(state.guidance?.key==='review'?state.readiness:null);
    $('captureAttemptStatus').hidden=true;
    $('recordingNotice').hidden=true;
    $('guidePanel').dataset.placement=String(!!placement);
    for(const eye of ['left','right']){
        const counts=placement?.visibility_counts;
        const missing=!!(counts&&Number.isFinite(counts[eye])&&counts[eye]<counts.expected);
        $(`${eye}EyeLabel`).dataset.missing=String(missing);
        text(`${eye}EyeLabel`,`${eye.toUpperCase()} CAMERA${missing?` · SHOW FULL BOARD (${counts[eye]}/${counts.expected})`:''}`);
    }
    if(!review)return;
    const checking=state.stage==='checking';
    const error=checking?state.report?.max_rms_px:review.fit_max_rms_px;
    const limit=review.threshold_px??state.policy?.check_reprojection_rms_px;
    const metric=Number.isFinite(error)&&Number.isFinite(limit)&&error>limit
        ?`${checking?'Validation':'Fit'} error ${error.toFixed(1)} px exceeds the ${limit.toFixed(1)} px limit.`
        :review.reason||'Saved poses do not agree with the camera images.';
    if(placement){
        const available=!guidanceHidden(placement.key);
        text('guideEyebrow','POSITIONING GUIDE · YOU MOVE THE ARM');
        text('guideTitle',placement.at_target?'Position matched':available
            ?placement.adjustment?.kind==='move'?'Move into the blue area':'Follow the small tilt below'
            :placement.title||'Show the board clearly');
        text('guideInstruction',available
            ?placement.at_target?'Keep this position. Automatic recording is still paused.'
            :placement.adjustment?.kind==='move'?'Bring the board centre into the blue area in the left camera view, then follow the tilt.'
            :'Keep the board centre in the blue area. Follow the marked edge toward or away from the camera.'
            :placement.instruction);
        if(available)text('positionLegendText','Blue area: keep the board centre here · follow the tilt above');
        text('guidePausedLabel','Waiting for view');
        text('guidePausedHint','No fresh direction');
        $('recordingNotice').hidden=false;
        text('recordingNoticeText',`${metric} Positioning alone does not resolve this error.`);
        $('guideNext').hidden=!available&&!placement.frame_id;
        text('guideNext','Another suggestion');
        text('poseLegend','Red: measured start · Green: suggested pose · keep clear of table and camera rod');
    }else{
        text('guideEyebrow','CALIBRATION · PAUSED');
        text('guideTitle','Saved poses do not agree');
        text('guideInstruction',`${metric} Check the board mounting and measured arm pairing before continuing.`);
        text('guidePausedLabel','Accuracy check');
        text('guidePausedHint','Movement guide paused');
        $('guideNext').hidden=true;
    }
    text('badge','Needs review');
    $('reviewDetails').hidden=false;
    $('autoCapture').hidden=true;
    $('holdArea').hidden=true;
    $('holdArea').classList.toggle('captured',false);
    $('holdFill').style.width='0%';
    $('retryCapture').hidden=false;
    $('retryCapture').disabled=busy||!state.can_retry_capture||!!state.pending||!!state.assessing;
    text('retryCapture',state.pending?'Waiting for a diagnostic pose…':'Capture one diagnostic pose');
    // Novelty is still checked live, but a paused model must not invite the
    // operator to keep moving as though guided calibration were progressing.
    if(!state.pending&&!state.just_captured&&!state.error
        &&state.measurement?.reason?.startsWith('Pose already captured.')){
        text('message',placement?'This pose is already saved. The positioning guide is available; automatic recording remains paused.'
            :'This pose is already saved. Review the calibration issue above before continuing.');
    }
    const issue=state.capture_issue;
    let attempt='';
    if(state.pending)attempt=`Diagnostic capture waiting: ${state.capture_wait_reason||state.measurement?.reason||'show the full board and hold still.'}`;
    else if(state.just_captured||issue?.status==='resolved')attempt='Pose saved. Calibration was rechecked and still needs review.';
    else if(issue?.kind==='capture_timeout')attempt='Last retry timed out; no pose was saved.';
    else if(issue&&['capture_write_failed','capture_failed','capture_rejected'].includes(issue.kind))attempt=issue.message;
    text('captureAttemptStatus',attempt);
    $('captureAttemptStatus').hidden=!attempt;
}

// Recoverable failures wait through the normal capture gates, with a stop action.
// Accuracy and recording faults remain visible for review.
function renderCaptureIssue(state,busy){
    const issue=state.capture_issue;
    const active=issue&&issue.status!=='resolved';
    const pending=active&&issue.status==='retry_pending';
    const automatic=active&&issue.status==='retry_waiting'&&state.auto_capture;
    const resumable=!!(active&&issue.retryable&&!state.auto_capture&&!pending);
    $('retryCapture').hidden=!active||automatic||resumable;
    $('retryCapture').disabled=busy||!state.can_retry_capture||!!state.pending||!!state.assessing;
    text('retryCapture',pending?'Waiting for pose…':'Retry capture');
    const failures=state.failed_training_ids?.length||0;
    $('excludeFailed').hidden=!failures||state.stage!=='training';
    $('excludeFailed').disabled=busy||!state.can_exclude_failed_training;
    text('excludeFailed',`Exclude ${failures} rejected poses`);
    $('guidePanel').dataset.retry=active?issue.status:'';
    if(!active)return;
    if(automatic){
        if(state.guidance?.key==='unavailable')$('guideNext').hidden=true;
        text('guideEyebrow','CAPTURE · AUTOMATIC RETRY');
        text('guideTitle',state.hold_progress>0?'Hold still — retrying':'Will retry automatically');
        text('guideInstruction',issue.message);
        text('badge',state.hold_progress>0?'Hold steady':'Waiting to retry');
        $('holdArea').hidden=!!state.assessing;
        $('holdArea').classList.toggle('captured',false);
        $('holdFill').style.width=`${(state.hold_progress||0)*100}%`;
        text('holdLabel',state.assessing?'Checking saved poses':!state.feedback.connected?'Waiting for measured arm feedback':!state.capture_ready?'Waiting for a clear, steady pose':'Valid pose — hold still to retry');
        text('holdPercent',state.hold_progress>0?`${Math.round(state.hold_progress*100)}%`:'');
        $('autoCapture').hidden=false;
        $('autoCapture').disabled=busy;
        text('autoCapture','Stop automatic capture');
        return;
    }
    text('guideEyebrow',pending?'CAPTURE · RETRYING':'CAPTURE · RETRY NEEDED');
    text('guideTitle',resumable?'Automatic retry paused':pending?'Hold a clear, steady pose':issue.kind==='training_excluded'?'Rejected poses removed':'Retry this capture');
    text('guideInstruction',issue.message);
    $('guideNext').hidden=true;
    $('reviewDetails').hidden=false;
    text('badge',resumable?'Paused':pending?'Retrying…':'Retry needed');
    $('holdArea').hidden=true;
    $('holdArea').classList.toggle('captured',false);
    $('holdFill').style.width='0%';
    $('autoCapture').disabled=!resumable||busy||!!state.assessing;
    $('autoCapture').hidden=!resumable;
    if(resumable)text('autoCapture','Resume automatic capture');
}
