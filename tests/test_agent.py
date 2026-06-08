import pytest
from src.agent.cloud_agent import parse_json_from_text

def test_json_parsing_from_agent():
    # 1. Clean JSON
    raw_1 = '{"must_link": [[0, 1]], "cannot_link": [[2, 3]]}'
    res_1 = parse_json_from_text(raw_1)
    assert res_1 == {"must_link": [[0, 1]], "cannot_link": [[2, 3]]}
    
    # 2. Markdown wrapped JSON
    raw_2 = '```json\n{"must_link": [[1, 2]], "cannot_link": []}\n```'
    res_2 = parse_json_from_text(raw_2)
    assert res_2 == {"must_link": [[1, 2]], "cannot_link": []}
    
    # 3. Text around JSON
    raw_3 = 'Here is the result:\n{\n  "must_link": [[0, 2]],\n  "cannot_link": [[1, 3]]\n}\nHope this helps!'
    res_3 = parse_json_from_text(raw_3)
    assert res_3 == {"must_link": [[0, 2]], "cannot_link": [[1, 3]]}
    
    # 4. Invalid JSON
    raw_4 = 'Not a JSON at all'
    res_4 = parse_json_from_text(raw_4)
    assert res_4 == {"must_link": [], "cannot_link": []}


def test_github_models_agent_init():
    from src.agent.cloud_agent import GitHubModelsAgent
    import os
    
    agent = GitHubModelsAgent(api_key="ghp_test_token", model_name="gpt-4o-mini")
    assert agent.model_name == "gpt-4o-mini"
    assert agent.client.api_key == "ghp_test_token"
    assert str(agent.client.base_url).rstrip("/") == "https://models.inference.ai.azure.com"

    old_env = os.environ.get("GITHUB_TOKEN")
    if "GITHUB_TOKEN" in os.environ:
        del os.environ["GITHUB_TOKEN"]
    
    with pytest.raises(ValueError, match="GitHub Token is missing"):
        GitHubModelsAgent(api_key="")
        
    if old_env is not None:
        os.environ["GITHUB_TOKEN"] = old_env
