import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('target_judge',Path(__file__).parents[1]/'toolkit/bfs_targeted_judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def grade(**kw):
 return {**dict.fromkeys(judge.FIELDS,1.),'verdict':'correct',**kw}

@pytest.mark.parametrize('verdict',['wrong_target','no_swap','mixed'])
def test_high_visual_quality_cannot_reward_wrong_selection(verdict):
 assert judge.reward(grade(verdict=verdict))==0

@pytest.mark.parametrize('data',[{'verdict':'uncertain'},{'confidence':.69}])
def test_uncertainty_is_not_a_positive_training_label(data):
 assert judge.reward(grade(**data)) is None

def test_preservation_limits_reward_even_with_perfect_replacement():
 assert judge.reward(grade(other_people_preserved=.2))==2

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,-.1,1.1])
def test_malformed_judge_scores_cannot_reach_optimizer(bad):
 with pytest.raises(ValueError):judge.reward(grade(target_selection=bad))
