# DeepSeek API Models and Pricing

## Available Models

DeepSeek offers two main models accessible through their OpenAI-compatible API:

### deepseek-v4-flash

The cost-effective model supporting both thinking and non-thinking modes.

- **Context Length**: 1M tokens
- **Max Output**: 384K tokens
- **Features**: JSON Output, Tool Calls, Prefix Completion, FIM
- **Concurrency Limit**: 2500 requests

### deepseek-v4-pro

The high-performance reasoning model for complex tasks.

- **Context Length**: 1M tokens
- **Max Output**: 384K tokens
- **Features**: JSON Output, Tool Calls, Prefix Completion
- **Concurrency Limit**: 500 requests

## Pricing (per million tokens)

| Model | Input (Cache Miss) | Input (Cache Hit) | Output |
|-------|-------------------|-------------------|--------|
| deepseek-v4-flash | ¥1 | ¥0.02 | ¥2 |
| deepseek-v4-pro | ¥3 | ¥0.025 | ¥6 |

## Model Migration

The legacy model names were deprecated on July 24, 2026:

- `deepseek-chat` → `deepseek-v4-flash` (non-thinking mode)
- `deepseek-reasoner` → `deepseek-v4-flash` (thinking mode)

The `deepseek-v4-flash` model supports toggling between thinking and non-thinking modes via API parameters.

## API Compatibility

DeepSeek API is fully OpenAI-compatible:

- **Base URL**: `https://api.deepseek.com`
- **Anthropic Format**: `https://api.deepseek.com/anthropic`
- **SDK**: Any OpenAI-compatible client library works

## Thinking Mode

Thinking mode enables the model to show its reasoning process before providing an answer. This is useful for:

- Complex mathematical problems
- Multi-step reasoning tasks
- Code generation with explanation
- Scientific analysis

The reasoning content is returned in the `reasoning_content` field of the API response.

## Cost Optimization Strategies

1. **Prompt Caching**: DeepSeek automatically caches prompts. Repeated similar prompts benefit from cache hit pricing (¥0.02/M vs ¥1/M).

2. **Model Selection**: Use `deepseek-v4-flash` for most tasks. Only switch to `deepseek-v4-pro` for complex reasoning.

3. **Batch Processing**: Process multiple requests in parallel within concurrency limits.

4. **Context Window Management**: Trim unnecessary context to reduce input token costs.

5. **Off-peak Usage**: DeepSeek offers off-peak pricing during low-demand hours.
