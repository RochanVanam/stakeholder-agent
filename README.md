# GTM Stakeholder Agent

TypeScript implementation of the Stakeholder Agent for the GTM Agent Broker hackathon project.

The agent accepts an org chart and product context, researches the target account with GPT web search, maps public evidence back to org-chart roles, identifies a buying committee, and drafts first-touch outreach.

## Demo

```bash
export OPENAI_API_KEY="..."
npm run demo
```

`npm run demo` calls GPT through the OpenAI API. ChatGPT Plus does not automatically provide API access, so this project expects an API key in `OPENAI_API_KEY`.

The run is a multi-step agent workflow:

1. `planResearch()` decides what evidence to look for.
2. `conductCompanyResearch()` uses OpenAI web search to gather current public account signals.
3. `classifyStakeholders()` maps research evidence to the org chart.
4. `draftOutreach()` writes short, researched emails for the top contacts.

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
- `planResearch()`
- `conductCompanyResearch()`
- `classifyStakeholders()`
- `draftOutreach()`
- `getPathToRoot()`
- `runStakeholderAgent()`

`runStakeholderAgent()` is the intended entrypoint for the future Master GTM Agent.
