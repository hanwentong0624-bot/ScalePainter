import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_evaluation', Path(__file__).parents[1]/'scripts/prepare_evaluation.py')
utility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utility)


class EvaluationManifestTests(unittest.TestCase):
    def test_missing_outputs_remain_in_population_and_prompts_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = dict(id='9000', dataset='EditBench', seed=9000, prompt='generation prompt',
                        global_prompt='original global', local_prompt='original local',
                        source='reference.png', mask='mask.png')
            cases = root/'cases.jsonl'
            cases.write_text(json.dumps(case)+'\n')
            run = root/'run'
            run.mkdir()
            plan = dict(dry_run=False, cases=[case], variants=['A0','A_all'], expected_generations=2)
            (run/'plan.json').write_text(json.dumps(plan))
            rows = utility.build_rows(cases, run, root/'evaluation.jsonl')
            self.assertEqual(len(rows), 2)
            self.assertEqual([r['generation_status'] for r in rows], ['missing','missing'])
            self.assertEqual(rows[0]['global_prompt'], 'original global')
            self.assertEqual(rows[0]['local_prompt'], 'original local')
            self.assertEqual(rows[0]['raw'], 'run/9000/A0/raw.png')
            plan['cases'][0]['seed'] = 1
            (run/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'seed'):
                utility.build_rows(cases, run, root/'evaluation.jsonl')


if __name__ == '__main__':
    unittest.main()
