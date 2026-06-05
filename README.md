# GTM Stakeholder Agent

TypeScript implementation of the Stakeholder Agent for the Marketing Agent project.

The agent accepts an org chart and product context, asks GPT to score and classify the buying committee, and returns structured JSON for a future Master GTM Agent.

## Demo

```bash
export OPENAI_API_KEY="..."
npm run demo
```

`npm run demo` calls GPT through the OpenAI API. ChatGPT Plus does not automatically provide API access, so this project expects an API key in `OPENAI_API_KEY`.

The demo uses `example.json` and this context:

```json
{
  "sellerCompany": "Vercel",
  "targetCompany": "NVIDIA",
  "productCategory": "frontend hosting/deployment",
  "competitor": "Cloudflare"
}
```

## Public API

The main module exports:

- `loadOrgChart()`
- `flattenOrgTree()`
- `scorePerson()`
- `classifyStakeholders()`
- `getPathToRoot()`
- `runStakeholderAgent()`

`runStakeholderAgent()` is the intended entrypoint for the future Master GTM Agent.
