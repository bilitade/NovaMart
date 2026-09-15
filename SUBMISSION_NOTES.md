# Submission Notes

Score: 120/120 (100%). Screenshot: `screenshots/testscore.png`

## 1. Deployed on my personal AWS account, not the Udacity sandbox

The Udacity sandbox denies `bedrock-agentcore:*` entirely, so `CreateAgentRuntime` fails there (AccessDeniedException, see `screenshots/udacity_cloud_resource_access_denied_ agent_runtime.png`). Checked this via boto3, the CLI, and the Console. The full stack (guardrail, runtime, memory, all three Knowledge Bases) is deployed and verified on my own account instead.

## 2. Claude models aren't available on this account

My AWS account is registered in Ethiopia. Anthropic's API already supports the country, but Bedrock hasn't enabled Claude model access for this account yet, so every Claude call gets rejected. I used `openai.gpt-oss-20b` for the orchestrator and `openai.gpt-oss-120b` for the workers as an alternative.

`test_2_7_routing_uses_different_models` originally only matched the strings "haiku"/"sonnet", so it failed even though the model selection pattern itself is correct. I extended that check to also accept `gpt-oss-20b`/`gpt-oss-120b`, since the point of the test is checking the routing/reasoning split, not the vendor name.

## 3. Observability

`configure_observability()` builds a `loggingConfiguration` dict (CloudWatch log group at INFO level, X-Ray at 100% sampling) and passes it to `apply_observability_config()` in `src/agent_observability.py`, wrapped in try/except. That call turns on CloudWatch Transaction Search and sets the runtime's logging environment variables.

Separately, `agent_orchestrator.py` has its own X-Ray tracing (segments opened inside the routing tools) that produces the Service Map. Screenshot: `screenshots/xray_service_map.png`.
