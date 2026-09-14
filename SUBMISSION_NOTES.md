# Submission Notes

Score: 110/120 (92%), see `screenshots/testscore.png`

## 1. Personal AWS account

The Udacity sandbox account's IAM policy denies all `bedrock-agentcore:*` permissions, so `CreateAgentRuntime` fails there. Confirmed via boto3, CLI, and the Console. I redeployed the full stack on my own AWS account, where the guardrail, runtime, memory, and all three Knowledge Bases are live and verified.

## 2. Claude models geo-blocked (10 pts lost)

My AWS account is registered in Ethiopia, which isn't on Anthropic's supported-country list, so every Claude call is rejected by Bedrock. I used `openai.gpt-oss-20b` (orchestrator) and `openai.gpt-oss-120b` (workers) instead, tested and working end to end. The 10 points lost are from `test_2_7`, which checks the model ID literally contains "haiku"/"sonnet" — only real Claude access can satisfy that.

## 3. X-Ray Service Map

`agent_observability.py`, referenced in the project docs, isn't part of this starter, so no trace data was being emitted anywhere. I implemented it myself using the official AWS docs: enabled CloudWatch Transaction Search account-wide and added X-Ray segments inside the routing tools. I also found and fixed a bug in the starter's compatibility patch (it mocked the right API on the wrong boto3 service), which brought Task 6 to 20/20. Result is a real, working Service Map, see `screenshots/xray_service_map.png`.
