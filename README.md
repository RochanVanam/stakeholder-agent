# Contact Agent

Tool-using agent that researches who to contact, then spends Hunter credits only on those selected people.

The agent flow:

1. Normalizes the company people JSON.
2. Locally ranks plausible contact candidates so the model sees a focused shortlist.
3. Uses OpenAI to plan what a human researcher should investigate.
4. Uses OpenAI web search to gather public company/product/stakeholder signals.
5. Uses OpenAI again to select the top contacts from the shortlist using the research evidence.
6. Calls Hunter Email Finder only for those selected contacts and writes a new JSON file.

It does not email everyone, guess emails, return personal emails, phone numbers, scraped emails, or pick CEOs/founders when reachable operator contacts exist.

## Demo

```bash
export OPENAI_API_KEY="..."
export HUNTER_API_KEY="..."
npm run demo
```

The demo fetches people from:

```text
https://shiptheagent.vercel.app/api/people?domain=coinbase
```

Then it selects the top 3 people for the Vercel/frontend deployment context, calls Hunter only for those 3 people, and writes:

```text
coinbase-org-context-with-emails.json
```

For a different run:

```bash
node --experimental-strip-types src/index.ts \
  --domain coinbase \
  --output coinbase-org-context-with-emails.json \
  --top 3 \
  --sellerCompany Vercel \
  --targetCompany Coinbase \
  --productCategory "frontend hosting/deployment" \
  --competitor Cloudflare
```

## Public API

Main entrypoint:

```ts
runContactAgent(input, context, options)
```

Useful lower-level tools:

- `rankContactCandidates()`
- `planTargetResearch()`
- `conductTargetResearch()`
- `findCompanyEmail()`
- `runEmailFinderForPeople()`
- `addSelectedContactsToPeopleInput()`

## Output

The CLI returns selected contacts only:

```json
{
  "company": {},
  "selectedContacts": [],
  "contactAgent": {}
}
```

Each selected contact is annotated with:

- `contactPriority`
- `contactScore`
- `contactReasons`
- `contactEvidenceUrls`
- `email`
- `emailStatus`

It also adds a top-level `contactAgent` summary with selected contact IDs, research signals/sources, and Hunter lookup counts.
