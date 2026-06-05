import test from "node:test";
import assert from "node:assert/strict";
import {
  addSelectedContactsToPeopleInput,
  findCompanyEmail,
  loadPeopleInputFromApi,
  normalizePeopleInput,
  rankContactCandidates,
  runContactAgent
} from "./index.ts";

test("normalizes the Coinbase org-context schema", () => {
  const people = normalizePeopleInput({
    company: { name: "Coinbase", domain: "coinbase.com" },
    people: [{ name: "Alesia Haas", title: "Chief Financial Officer", valueTier: "executive" }]
  });

  assert.deepEqual(people, [
    {
      inputId: "0",
      firstName: "Alesia",
      lastName: "Haas",
      fullName: "Alesia Haas",
      title: "Chief Financial Officer",
      department: undefined,
      companyName: "Coinbase",
      companyDomain: "coinbase.com",
      linkedinUrl: undefined,
      valueTier: "executive",
      influence: undefined,
      seniority: undefined,
      seniorityRank: undefined,
      reportSpan: undefined,
      directReports: undefined,
      reportsTo: null,
      level: null,
      teams: undefined
    }
  ]);
});

test("ranks decision makers above assistants and team roster entries", () => {
  const people = normalizePeopleInput({
    company: { name: "Coinbase", domain: "coinbase.com" },
    people: [
      {
        name: "Nikki H.",
        title: "Senior Executive Business Partner to CEO",
        valueTier: "executive",
        influence: "none"
      },
      {
        name: "Brian Armstrong",
        title: "CEO & Co-founder",
        valueTier: "executive",
        influence: "very-high",
        reportSpan: 805
      },
      {
        name: "Jesse Pollak",
        title: "VP, Engineering",
        department: "Engineering",
        valueTier: "senior-leader",
        influence: "low"
      },
      {
        name: "Aadhav Sundar",
        title: null,
        department: "Software Development",
        valueTier: "team-member",
        influence: "none"
      }
    ]
  });

  const ranked = rankContactCandidates(people, { productCategory: "frontend hosting/deployment" });

  assert.equal(ranked[0]?.fullName, "Jesse Pollak");
  assert.equal(ranked.some((person) => person.fullName === "Brian Armstrong"), false);
  assert.equal(ranked.some((person) => person.fullName === "Aadhav Sundar"), false);
});

test("loads people from the API endpoint and fills company defaults", async () => {
  const input = await loadPeopleInputFromApi(
    "coinbase",
    mockFetch([
      {
        data: {
          people: [{ name: "Jane Doe", title: "Director, Platform Engineering" }]
        }
      }
    ])
  );
  const people = normalizePeopleInput(input);

  assert.equal(people[0]?.fullName, "Jane Doe");
  assert.equal(people[0]?.companyName, "Coinbase");
  assert.equal(people[0]?.companyDomain, "coinbase.com");
});

test("returns found when Hunter Email Finder returns a work email", async () => {
  const fetchImpl = mockFetch([
    {
      data: {
        first_name: "Alesia",
        last_name: "Haas",
        email: "alesia@coinbase.com",
        score: 96,
        verification: { status: "valid" }
      }
    }
  ]);

  const result = await findCompanyEmail(
    {
      fullName: "Alesia Haas",
      firstName: "Alesia",
      lastName: "Haas",
      title: "Chief Financial Officer",
      companyName: "Coinbase",
      companyDomain: "coinbase.com"
    },
    { hunterApiKey: "test-key", fetchImpl }
  );

  assert.equal(result.status, "found");
  assert.equal(result.email, "alesia@coinbase.com");
  assert.equal(result.confidence, "high");
  assert.equal(result.provider, "hunter");
  assert.equal(result.matchMethod, "email_finder");
});

test("runs selector then Hunter only for selected top contacts", async () => {
  const input = {
    company: { name: "Coinbase", domain: "coinbase.com" },
    people: [
      {
        name: "Brian Armstrong",
        title: "CEO & Co-founder",
        valueTier: "executive",
        influence: "very-high",
        reportSpan: 805
      },
      {
        name: "Jesse Pollak",
        title: "VP, Engineering",
        department: "Engineering",
        valueTier: "senior-leader",
        influence: "low",
        reportSpan: 7
      },
      {
        name: "Nikki H.",
        title: "Senior Executive Business Partner to CEO",
        valueTier: "executive",
        influence: "none"
      }
    ]
  };
  const fetchImpl = mockFetch([
    {
      output_text: JSON.stringify({
        researchQuestions: ["Who owns developer platform at Coinbase?"],
        searchQueries: ["Coinbase developer platform engineering leadership"],
        evidenceNeeded: ["Public evidence about engineering platform ownership."]
      })
    },
    {
      output_text: JSON.stringify({
        companySignals: ["Coinbase has public engineering platform needs."],
        productFitSignals: ["Frontend deployment is relevant to engineering leadership."],
        competitorSignals: [],
        stakeholderSignals: [
          {
            topic: "Engineering leadership",
            evidence: "Jesse Pollak is a stronger technical buyer than an executive assistant.",
            sourceUrl: "https://example.com/coinbase-engineering",
            relevantRoles: ["VP Engineering"],
            relevantPeople: ["Jesse Pollak"]
          }
        ],
        recommendedContactAngles: ["Lead with developer platform velocity."],
        sources: [{ title: "Coinbase Engineering", url: "https://example.com/coinbase-engineering" }]
      })
    },
    {
      output_text: JSON.stringify({
        selectedContacts: [
          {
            inputId: "1",
            fullName: "Jesse Pollak",
            contactScore: 91,
            contactReasons: ["Engineering buyer for developer platform decisions."],
            evidenceUrls: ["https://example.com/coinbase-engineering"]
          }
        ]
      })
    },
    {
      data: {
        first_name: "Jesse",
        last_name: "Pollak",
        email: "jesse@coinbase.com",
        score: 92,
        verification: { status: "valid" }
      }
    }
  ]);

  const result = await runContactAgent(
    input,
    { productCategory: "frontend hosting/deployment" },
    {
      openaiApiKey: "openai-test-key",
      hunterApiKey: "hunter-test-key",
      fetchImpl,
      topN: 1
    }
  );
  const enriched = result.enrichedJson as { selectedContacts: Array<Record<string, unknown>>; people?: unknown[] };

  assert.equal(result.selectedContacts.length, 1);
  assert.equal(result.selectedContacts[0]?.fullName, "Jesse Pollak");
  assert.deepEqual(result.selectedContacts[0]?.evidenceUrls, ["https://example.com/coinbase-engineering"]);
  assert.equal(result.emailLookup.summary.total, 1);
  assert.equal(enriched.selectedContacts.length, 1);
  assert.equal(enriched.selectedContacts[0]?.name, "Jesse Pollak");
  assert.equal(enriched.selectedContacts[0]?.email, "jesse@coinbase.com");
  assert.equal(enriched.selectedContacts[0]?.contactPriority, 1);
  assert.deepEqual(enriched.selectedContacts[0]?.contactEvidenceUrls, ["https://example.com/coinbase-engineering"]);
  assert.equal(enriched.people, undefined);
});

test("does not accept a CEO selection when a realistic operator exists", async () => {
  const input = {
    company: { name: "Coinbase", domain: "coinbase.com" },
    people: [
      {
        name: "Brian Armstrong",
        title: "CEO & Co-founder",
        valueTier: "executive",
        influence: "very-high",
        reportSpan: 805
      },
      {
        name: "Jane Doe",
        title: "Director, Platform Engineering",
        department: "Engineering",
        valueTier: "leader",
        influence: "high"
      }
    ]
  };
  const fetchImpl = mockFetch([
    {
      output_text: JSON.stringify({
        researchQuestions: ["Who owns platform engineering?"],
        searchQueries: ["Coinbase platform engineering director"],
        evidenceNeeded: ["Operator ownership."]
      })
    },
    {
      output_text: JSON.stringify({
        companySignals: [],
        productFitSignals: [],
        competitorSignals: [],
        stakeholderSignals: [],
        recommendedContactAngles: [],
        sources: []
      })
    },
    {
      output_text: JSON.stringify({
        selectedContacts: [
          {
            inputId: "0",
            fullName: "Brian Armstrong",
            contactScore: 99,
            contactReasons: ["Highest authority."]
          }
        ]
      })
    },
    {
      data: {
        first_name: "Jane",
        last_name: "Doe",
        email: "jane@coinbase.com",
        score: 90,
        verification: { status: "valid" }
      }
    }
  ]);

  const result = await runContactAgent(
    input,
    { productCategory: "frontend hosting/deployment" },
    {
      openaiApiKey: "openai-test-key",
      hunterApiKey: "hunter-test-key",
      fetchImpl,
      topN: 1
    }
  );

  assert.equal(result.selectedContacts[0]?.fullName, "Jane Doe");
  assert.equal(result.emailLookup.results[0]?.email, "jane@coinbase.com");
});

test("adds selection metadata and emails back onto selected people only", () => {
  const input = {
    company: { name: "Coinbase", domain: "coinbase.com" },
    people: [
      { name: "Alesia Haas", title: "Chief Financial Officer" },
      { name: "Jane Doe", title: "VP Engineering" }
    ]
  };

  const enriched = addSelectedContactsToPeopleInput(
    input,
    [
      {
        inputId: "0",
        firstName: "Alesia",
        lastName: "Haas",
        fullName: "Alesia Haas",
        title: "Chief Financial Officer",
        companyName: "Coinbase",
        companyDomain: "coinbase.com",
        contactPriority: 1,
        contactScore: 95,
        contactReasons: ["Owns finance approval."],
        evidenceUrls: ["https://example.com/alesia"]
      }
    ],
    {
      results: [
        {
          inputId: "0",
          fullName: "Alesia Haas",
          companyName: "Coinbase",
          companyDomain: "coinbase.com",
          email: "alesia@coinbase.com",
          status: "found",
          confidence: "high",
          provider: "hunter",
          matchMethod: "email_finder",
          notes: []
        }
      ],
      summary: {
        total: 1,
        found: 1,
        notFound: 0,
        ambiguous: 0,
        errors: 0
      }
    }
  ) as { selectedContacts: Array<Record<string, unknown>>; people?: unknown[]; contactAgent: Record<string, unknown> };

  assert.equal(enriched.selectedContacts.length, 1);
  assert.equal(enriched.selectedContacts[0]?.email, "alesia@coinbase.com");
  assert.equal(enriched.selectedContacts[0]?.contactScore, 95);
  assert.deepEqual(enriched.selectedContacts[0]?.contactEvidenceUrls, ["https://example.com/alesia"]);
  assert.equal(enriched.people, undefined);
  assert.deepEqual(enriched.contactAgent.selectedContactIds, ["0"]);
});

function mockFetch(payloads: unknown[], statuses: number[] = []): typeof fetch {
  let index = 0;

  return (async () => {
    const payload = payloads[index];
    const status = statuses[index] ?? 200;
    index += 1;

    return new Response(JSON.stringify(payload), {
      status,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;
}
