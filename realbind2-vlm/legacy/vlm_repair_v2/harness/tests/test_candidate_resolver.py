import unittest
from dataclasses import replace
from harness.referents.candidate_resolver import CandidateResolver, center_x, same_region
from harness.referents.referent_resolver import resolve_question
from harness.referents.referent_spec import Provenanced, Provenance
from harness.planner.predicate_dag import build_predicate_dag

QUESTION='Between the cat and the dog,\nwhat color is the one on the left?\nAnswer with only the color.'
def candidate(ref,idx,box,score=0.5):
    return {'candidate_id':f'{ref}:{idx}','bbox_xyxy_pixels':list(box),'source_backend':'grounding_dino','native_grounding_score':score,'siglip_diagnostic':{'z_raw':score}}
def setup(relation=None):
    refs=resolve_question(QUESTION,'REL2C_LEFT'); dag=build_predicate_dag(QUESTION,'REL2C_LEFT',refs)
    specs=refs.specs
    if relation:
        specs=(replace(specs[0],relation_constraints=(Provenanced(relation,Provenance.QUESTION_TEXT,'explicit asserted relation'),)),specs[1])
    return dag,specs
def run(a,b,relation=None,state=None):
    dag,specs=setup(relation)
    return CandidateResolver().resolve(dag,specs,{'A':a,'B':b},state or {},family_id='test',run_id='unit')
class TestCandidateResolver(unittest.TestCase):
    def test_selection_side_is_not_constraint(self):
        r=run([candidate('A',0,(30,0,40,10))],[candidate('B',0,(0,0,10,10))])
        self.assertEqual(r.status,'RESOLVED_UNIQUE')
        self.assertEqual(r.constraints,('DISTINCT_EXACT_REGION',))
    def test_both_asserted_directions(self):
        a=[candidate('A',0,(0,0,10,10)),candidate('A',1,(30,0,40,10))]
        b=[candidate('B',0,(20,0,25,10))]
        left=run(a,b,'LEFT_OF(A,B)'); right=run(a,b,'RIGHT_OF(A,B)')
        self.assertEqual(left.selected['A'],'A:0'); self.assertEqual(right.selected['A'],'A:1')
        self.assertEqual(left.valid_count,1); self.assertEqual(right.valid_count,1)
    def test_tie_and_near_tie_strict(self):
        a=[candidate('A',0,(0,0,10,10))]
        self.assertEqual(run(a,[candidate('B',0,(0,20,10,30))],'LEFT_OF(A,B)').status,'ABSTAIN_NO_VALID_ASSIGNMENT')
        self.assertEqual(run(a,[candidate('B',0,(0.000001,20,10.000001,30))],'LEFT_OF(A,B)').status,'RESOLVED_UNIQUE')
    def test_overlap_and_containment_use_centers(self):
        a=[candidate('A',0,(0,0,20,20))]
        self.assertEqual(run(a,[candidate('B',0,(5,5,15,15))],'LEFT_OF(A,B)').valid_count,0)
        self.assertEqual(run(a,[candidate('B',0,(9,0,21,20))],'LEFT_OF(A,B)').valid_count,1)
    def test_exact_region_collision_only(self):
        a=[candidate('A',0,(0,0,10,10))]
        self.assertEqual(run(a,[candidate('B',0,(0,0,10,10))]).valid_count,0)
        self.assertEqual(run(a,[candidate('B',0,(0,0,10.000001,10))]).valid_count,1)
    def test_scores_do_not_affect_decision(self):
        a=[candidate('A',0,(0,0,10,10),0.1),candidate('A',1,(20,0,30,10),0.9)]
        b=[candidate('B',0,(40,0,50,10))]
        one=run(a,b); two=run([dict(c,native_grounding_score=1-c['native_grounding_score'],siglip_diagnostic={'z_raw':1-c['siglip_diagnostic']['z_raw']}) for c in a],b)
        self.assertEqual(one.status,'ABSTAIN_AMBIGUOUS');self.assertEqual(one.status,two.status)
        self.assertEqual([h['valid'] for h in one.hypotheses],[h['valid'] for h in two.hypotheses])
        self.assertFalse(one.persistent_entities)
    def test_unique_state_uncertified_and_reuse(self):
        a=[candidate('A',0,(0,0,10,10))];b=[candidate('B',0,(30,0,40,10))]
        r=run(a,b,state={'A':{'entity_id':'E_previous','region':{'bbox_xyxy_pixels':[0,0,10,10]}}})
        self.assertEqual(r.persistent_entities['A']['entity_id'],'E_previous')
        self.assertEqual(r.persistent_entities['A']['evidence_status'],'RESOLVED_BUT_UNCERTIFIED')
        self.assertEqual(run(a,b,state={'A':{'region':{'bbox_xyxy_pixels':[1,0,10,10]}}}).valid_count,0)
    def test_gt_input_rejected(self):
        a=[candidate('A',0,(0,0,10,10))];a[0]['gt_bbox']=[0,0,10,10]
        with self.assertRaises((AssertionError,ValueError)): run(a,[candidate('B',0,(20,0,30,10))])
if __name__=='__main__': unittest.main()
