import { readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type EmailStatus = "found" | "not_found" | "ambiguous" | "error";
export type Confidence = "high" | "medium" | "low";
export type MatchMethod = "email_finder";

export type ContactContext = {
  sellerCompany?: string;
  targetCompany?: string;
  productCategory?: string;
  competitor?: string;
};

export type NormalizedPerson = {
  inputId?: string;
  firstName?: string;
  lastName?: string;
  fullName: string;
  title?: string;
  department?: string;
  companyName: string;
  companyDomain?: string;
  linkedinUrl?: string;
  valueTier?: string;
  influence?: string;
  seniority?: string;
  seniorityRank?: number;
  reportSpan?: number;
  directReports?: number;
  reportsTo?: string | null;
  level?: number | null;
  teams?: string[];
};

export type SelectedContact = NormalizedPerson & {
  contactPriority: number;
  contactScore: number;
  contactReasons: string[];
  evidenceUrls?: string[];
};

export type EmailFinderResultItem = {
  inputId?: string;
  fullName: string;
  companyName: string;
  companyDomain?: string;
  email?: string;
  status: EmailStatus;
  confidence: Confidence;
  provider: "hunter";
  matchMethod: MatchMethod;
  notes: string[];
};

export type EmailFinderResult = {
  results: EmailFinderResultItem[];
  summary: {
    total: number;
    found: number;
    notFound: number;
    ambiguous: number;
    errors: number;
  };
};

export type ResearchPlan = {
  researchQuestions: string[];
  searchQueries: string[];
  evidenceNeeded: string[];
};

export type ResearchSignal = {
  topic: string;
  evidence: string;
  sourceUrl?: string;
  relevantRoles: string[];
  relevantPeople?: string[];
};

export type ResearchReport = {
  companySignals: string[];
  productFitSignals: string[];
  competitorSignals: string[];
  stakeholderSignals: ResearchSignal[];
  recommendedContactAngles: string[];
  sources: Array<{
    title: string;
    url: string;
  }>;
};

export type ContactAgentTraceStep = {
  step: string;
  tool: "openai" | "hunter" | "local";
  outputSummary: string;
};

export type ContactAgentResult = {
  contactContext: ContactContext;
  researchPlan: ResearchPlan;
  researchReport: ResearchReport;
  selectedContacts: SelectedContact[];
  emailLookup: EmailFinderResult;
  enrichedJson: unknown;
  trace: ContactAgentTraceStep[];
};

export type ContactAgentOptions = {
  hunterApiKey?: string;
  openaiApiKey?: string;
  model?: string;
  topN?: number;
  candidateLimit?: number;
  fetchImpl?: typeof fetch;
};

type UnknownRecord = Record<string, unknown>;

type HunterEmailFinderResponse = {
  data?: {
    first_name?: string | null;
    last_name?: string | null;
    email?: string | null;
    score?: number | null;
    domain?: string | null;
    position?: string | null;
    company?: string | null;
    linkedin_url?: string | null;
    verification?: {
      status?: string | null;
      date?: string | null;
    } | null;
  } | null;
  errors?: Array<{
    id?: string;
    code?: number;
    details?: string;
  }>;
};

type OpenAIResponse = {
  output_text?: string;
  output?: Array<{
    content?: Array<{
      text?: string;
    }>;
  }>;
  error?: {
    message?: string;
  };
};

const HUNTER_EMAIL_FINDER_URL = "https://api.hunter.io/v2/email-finder";
const OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses";
const PEOPLE_API_URL = "https://shiptheagent.vercel.app/api/people";
const DEFAULT_SELECTOR_MODEL = "gpt-5-nano";

export function loadPeopleInput(path = "coinbase-org-context.json"): unknown {
  return JSON.parse(readFileSync(resolve(path), "utf8")) as unknown;
}

export async function loadPeopleInputFromApi(
  companyDomain: string,
  fetchImpl: typeof fetch = fetch
): Promise<unknown> {
  const url = new URL(PEOPLE_API_URL);
  url.searchParams.set("domain", companyDomain);

  const response = await fetchImpl(url.toString(), {
    method: "GET",
    headers: {
      accept: "application/json"
    }
  });
  const text = await response.text();
  const json = text ? (JSON.parse(text) as unknown) : {};
  if (!response.ok) {
    throw new Error(`People API error ${response.status}: ${response.statusText}`);
  }

  return normalizePeopleApiInput(json, companyDomain);
}

export function normalizePeopleInput(input: unknown): NormalizedPerson[] {
  const root = asRecord(input);
  const data = asRecord(root?.data);
  const company = asRecord(root?.company) ?? asRecord(data?.company);
  const defaultCompanyName = readString(company, ["name", "companyName", "organizationName"]);
  const defaultCompanyDomain = normalizeCompanyDomainValue(readString(company, ["domain", "companyDomain", "primaryDomain", "website"]));
  const people = getPeopleArray(input) ?? [];

  return people
    .map((rawPerson, index) => normalizePerson(rawPerson, index, defaultCompanyName, defaultCompanyDomain))
    .filter((person): person is NormalizedPerson => Boolean(person));
}

export async function runContactAgent(
  input: unknown,
  context: ContactContext = {},
  options: ContactAgentOptions = {}
): Promise<ContactAgentResult> {
  const people = normalizePeopleInput(input);
  const topN = options.topN ?? 3;
  const candidateLimit = options.candidateLimit ?? 30;
  const trace: ContactAgentTraceStep[] = [];

  const candidates = rankContactCandidates(people, context).slice(0, candidateLimit);
  trace.push({
    step: "rankContactCandidates",
    tool: "local",
    outputSummary: `Prepared ${candidates.length} candidate contacts from ${people.length} people.`
  });

  const researchPlan = await planTargetResearch(input, context, candidates, options);
  trace.push({
    step: "planTargetResearch",
    tool: "openai",
    outputSummary: `${researchPlan.searchQueries.length} search queries and ${researchPlan.evidenceNeeded.length} evidence targets.`
  });

  const researchReport = await conductTargetResearch(input, context, researchPlan, candidates, options);
  trace.push({
    step: "conductTargetResearch",
    tool: "openai",
    outputSummary: `${researchReport.stakeholderSignals.length} stakeholder signals from ${researchReport.sources.length} sources.`
  });

  const selectedContacts = await selectContactTargets(candidates, context, researchReport, { ...options, topN });
  trace.push({
    step: "selectContactTargets",
    tool: "openai",
    outputSummary: `Selected ${selectedContacts.length} researched contacts before email lookup.`
  });

  const emailLookup = await runEmailFinderForPeople(selectedContacts, options);
  trace.push({
    step: "findCompanyEmails",
    tool: "hunter",
    outputSummary: `Looked up ${emailLookup.summary.total} selected contacts; found ${emailLookup.summary.found} emails.`
  });

  const enrichedJson = addSelectedContactsToPeopleInput(input, selectedContacts, emailLookup, researchPlan, researchReport);
  trace.push({
    step: "writeEnrichedJson",
    tool: "local",
    outputSummary: "Added contact selection metadata and Hunter emails to selected people only."
  });

  return {
    contactContext: context,
    researchPlan,
    researchReport,
    selectedContacts,
    emailLookup,
    enrichedJson,
    trace
  };
}

export async function planTargetResearch(
  input: unknown,
  context: ContactContext,
  candidates: SelectedContact[],
  options: ContactAgentOptions = {}
): Promise<ResearchPlan> {
  const apiKey = options.openaiApiKey ?? getEnvValue("OPENAI_API_KEY");
  if (!apiKey) {
    return fallbackResearchPlan(input, context);
  }

  try {
    const model = options.model ?? getEnvValue("OPENAI_MODEL") ?? DEFAULT_SELECTOR_MODEL;
    const response = await callOpenAIJson<Partial<ResearchPlan>>({
      apiKey,
      model,
      fetchImpl: options.fetchImpl,
      maxOutputTokens: 1600,
      payload: {
        task: "Plan web research before choosing sales contacts.",
        contactContext: context,
        company: asRecord(input)?.company,
        candidateSample: candidates.slice(0, 12).map(toResearchCandidate),
        instructions:
          "Return researchQuestions, searchQueries, and evidenceNeeded. Focus on evidence a human GTM researcher would gather before deciding who to contact. Queries should target company engineering priorities, product/platform initiatives, relevant leadership, and competitor/tooling context."
      }
    });

    return sanitizeResearchPlan(response, input, context);
  } catch {
    return fallbackResearchPlan(input, context);
  }
}

export async function conductTargetResearch(
  input: unknown,
  context: ContactContext,
  researchPlan: ResearchPlan,
  candidates: SelectedContact[],
  options: ContactAgentOptions = {}
): Promise<ResearchReport> {
  const apiKey = options.openaiApiKey ?? getEnvValue("OPENAI_API_KEY");
  if (!apiKey) {
    return fallbackResearchReport();
  }

  try {
    const model = options.model ?? getEnvValue("OPENAI_MODEL") ?? DEFAULT_SELECTOR_MODEL;
    const response = await callOpenAIJson<Partial<ResearchReport>>({
      apiKey,
      model,
      fetchImpl: options.fetchImpl,
      useWebSearch: true,
      maxOutputTokens: 4200,
      payload: {
        task: "Use web search to research the target company before selecting contacts.",
        contactContext: context,
        company: asRecord(input)?.company,
        researchPlan,
        candidateSample: candidates.slice(0, 20).map(toResearchCandidate),
        instructions:
          "Search the web for current public evidence about the target company's priorities, engineering/product/platform initiatives, relevant teams, and competitor/tooling context. Return companySignals, productFitSignals, competitorSignals, stakeholderSignals, recommendedContactAngles, and sources. Map evidence to relevant roles and named candidates when possible. Do not invent private facts."
      }
    });

    return sanitizeResearchReport(response);
  } catch {
    return fallbackResearchReport();
  }
}

export async function runEmailFinderForPeople(
  people: NormalizedPerson[],
  options: ContactAgentOptions = {}
): Promise<EmailFinderResult> {
  const results: EmailFinderResultItem[] = [];

  for (const person of people) {
    results.push(await findCompanyEmail(person, options));
  }

  return {
    results,
    summary: summarizeResults(results)
  };
}

export async function findCompanyEmail(
  person: NormalizedPerson,
  options: ContactAgentOptions = {}
): Promise<EmailFinderResultItem> {
  const apiKey = options.hunterApiKey ?? getEnvValue("HUNTER_API_KEY");
  const baseResult = createBaseResult(person);

  if (!apiKey) {
    return {
      ...baseResult,
      status: "error",
      confidence: "low",
      matchMethod: "email_finder",
      notes: ["Missing HUNTER_API_KEY in environment or .env."]
    };
  }

  if (!person.companyDomain || !person.firstName || !person.lastName) {
    return {
      ...baseResult,
      status: "not_found",
      confidence: "low",
      matchMethod: "email_finder",
      notes: ["Hunter Email Finder requires company domain, first name, and last name."]
    };
  }

  try {
    const hunterResult = await callHunterEmailFinder(person, apiKey, options.fetchImpl);
    const emailResult = extractHunterWorkEmail(hunterResult, person.companyDomain);
    if (!emailResult) {
      return {
        ...baseResult,
        status: "not_found",
        confidence: "low",
        matchMethod: "email_finder",
        notes: ["Hunter did not return a work email for this person."]
      };
    }

    return {
      ...baseResult,
      email: emailResult.email,
      status: "found",
      confidence: confidenceFromHunter(hunterResult),
      matchMethod: "email_finder",
      notes: emailResult.notes
    };
  } catch (error) {
    if (isHunterNotFoundError(error)) {
      return {
        ...baseResult,
        status: "not_found",
        confidence: "low",
        matchMethod: "email_finder",
        notes: ["Hunter did not find an email for this person."]
      };
    }

    return {
      ...baseResult,
      status: "error",
      confidence: "low",
      matchMethod: "email_finder",
      notes: [error instanceof Error ? error.message : "Unknown Hunter lookup error."]
    };
  }
}

export function addSelectedContactsToPeopleInput(
  input: unknown,
  selectedContacts: SelectedContact[],
  emailLookup: EmailFinderResult,
  researchPlan?: ResearchPlan,
  researchReport?: ResearchReport
): unknown {
  const cloned = structuredCloneJson(input);
  const root = asRecord(cloned);
  const people = getPeopleArray(cloned) ?? [];

  const emailByInputId = new Map(emailLookup.results.map((result) => [result.inputId, result]));
  const selectedPeople: UnknownRecord[] = [];
  for (const contact of selectedContacts) {
    const index = findInputPersonIndex(people, contact.inputId);
    const originalPerson = index === -1 ? undefined : asRecord(people[index]);
    const person = originalPerson
      ? (structuredCloneJson(originalPerson) as UnknownRecord)
      : createPersonRecordFromContact(contact);
    const emailResult = emailByInputId.get(contact.inputId);
    if (person) {
      person["contactPriority"] = contact.contactPriority;
      person["contactScore"] = contact.contactScore;
      person["contactReasons"] = contact.contactReasons;
      if (contact.evidenceUrls && contact.evidenceUrls.length > 0) {
        person["contactEvidenceUrls"] = contact.evidenceUrls;
      }
      person["email"] = emailResult?.email ?? null;
      person["emailStatus"] = emailResult?.status ?? "not_found";
      selectedPeople.push(person);
    }
  }

  return {
    company: getCompanyRecord(root, selectedContacts),
    selectedContacts: selectedPeople,
    contactAgent: {
      selectedContactIds: selectedContacts.map((contact) => contact.inputId),
      emailSummary: emailLookup.summary,
      research: {
        questions: researchPlan?.researchQuestions ?? [],
        searchQueries: researchPlan?.searchQueries ?? [],
        signals: [
          ...(researchReport?.companySignals ?? []),
          ...(researchReport?.productFitSignals ?? []),
          ...(researchReport?.competitorSignals ?? [])
        ].slice(0, 12),
        sources: researchReport?.sources ?? []
      }
    }
  };
}

export function addEmailsToPeopleInput(input: unknown, result: EmailFinderResult): unknown {
  const cloned = structuredCloneJson(input);
  const people = getPeopleArray(cloned);
  if (!people) {
    return cloned;
  }

  for (const item of result.results) {
    const index = findInputPersonIndex(people, item.inputId);
    if (index === -1) {
      continue;
    }

    const person = asRecord(people[index]);
    if (person) {
      person["email"] = item.email ?? null;
    }
  }

  return cloned;
}

export function rankContactCandidates(people: NormalizedPerson[], context: ContactContext = {}): SelectedContact[] {
  return people
    .map((person) => {
      const scoring = scoreCandidate(person, context);
      return {
        ...person,
        contactPriority: 0,
        contactScore: scoring.score,
        contactReasons: scoring.reasons
      };
    })
    .filter((person) => person.contactScore > 0)
    .sort((a, b) => b.contactScore - a.contactScore || compareOptionalNumber(a.seniorityRank, b.seniorityRank))
    .map((person, index) => ({
      ...person,
      contactPriority: index + 1
    }));
}

function fallbackResearchPlan(input: unknown, context: ContactContext): ResearchPlan {
  const company = asRecord(asRecord(input)?.company);
  const targetCompany = context.targetCompany ?? readString(company, ["name", "companyName", "organizationName"]) ?? "target company";
  const productCategory = context.productCategory ?? "the product category";
  const competitor = context.competitor ?? "the current alternative";

  return {
    researchQuestions: [
      `What public priorities does ${targetCompany} have related to ${productCategory}?`,
      `Which teams or leaders at ${targetCompany} likely own ${productCategory} decisions?`,
      `What signals show ${targetCompany} might compare the seller against ${competitor}?`
    ],
    searchQueries: [
      `${targetCompany} ${productCategory} engineering priorities`,
      `${targetCompany} platform engineering developer experience infrastructure`,
      `${targetCompany} ${competitor} frontend deployment hosting`
    ],
    evidenceNeeded: [
      "Current public initiatives related to product fit.",
      "Roles or teams likely to own the buying decision.",
      "Named leaders connected to engineering, platform, developer experience, infrastructure, or web systems."
    ]
  };
}

function sanitizeResearchPlan(plan: Partial<ResearchPlan>, input: unknown, context: ContactContext): ResearchPlan {
  const fallback = fallbackResearchPlan(input, context);
  const researchQuestions = sanitizeStringArray(plan.researchQuestions).slice(0, 8);
  const searchQueries = sanitizeStringArray(plan.searchQueries).slice(0, 8);
  const evidenceNeeded = sanitizeStringArray(plan.evidenceNeeded).slice(0, 8);

  return {
    researchQuestions: researchQuestions.length > 0 ? researchQuestions : fallback.researchQuestions,
    searchQueries: searchQueries.length > 0 ? searchQueries : fallback.searchQueries,
    evidenceNeeded: evidenceNeeded.length > 0 ? evidenceNeeded : fallback.evidenceNeeded
  };
}

function fallbackResearchReport(): ResearchReport {
  return {
    companySignals: [],
    productFitSignals: [],
    competitorSignals: [],
    stakeholderSignals: [],
    recommendedContactAngles: [],
    sources: []
  };
}

function sanitizeResearchReport(report: Partial<ResearchReport>): ResearchReport {
  const stakeholderSignals = Array.isArray(report.stakeholderSignals)
    ? report.stakeholderSignals
        .map((rawSignal) => {
          const signal = asRecord(rawSignal);
          if (!signal) {
            return undefined;
          }

          return {
            topic: sanitizeString(signal.topic),
            evidence: sanitizeString(signal.evidence),
            sourceUrl: sanitizeString(signal.sourceUrl) || undefined,
            relevantRoles: sanitizeStringArray(signal.relevantRoles),
            relevantPeople: sanitizeStringArray(signal.relevantPeople)
          };
        })
        .filter(
          (signal): signal is ResearchSignal =>
            Boolean(signal?.topic || signal?.evidence || signal?.relevantRoles.length || signal?.relevantPeople?.length)
        )
        .slice(0, 12)
    : [];

  const sources = Array.isArray(report.sources)
    ? report.sources
        .map((rawSource) => {
          const source = asRecord(rawSource);
          if (!source) {
            return undefined;
          }

          return {
            title: sanitizeString(source.title) || sanitizeString(source.url),
            url: sanitizeString(source.url)
          };
        })
        .filter((source): source is { title: string; url: string } => Boolean(source?.url))
        .slice(0, 12)
    : [];

  return {
    companySignals: sanitizeStringArray(report.companySignals).slice(0, 12),
    productFitSignals: sanitizeStringArray(report.productFitSignals).slice(0, 12),
    competitorSignals: sanitizeStringArray(report.competitorSignals).slice(0, 12),
    stakeholderSignals,
    recommendedContactAngles: sanitizeStringArray(report.recommendedContactAngles).slice(0, 8),
    sources
  };
}

function toResearchCandidate(candidate: SelectedContact): UnknownRecord {
  return {
    inputId: candidate.inputId,
    fullName: candidate.fullName,
    title: candidate.title,
    department: candidate.department,
    valueTier: candidate.valueTier,
    influence: candidate.influence,
    seniority: candidate.seniority,
    reportSpan: candidate.reportSpan,
    directReports: candidate.directReports,
    reportsTo: candidate.reportsTo,
    localScore: candidate.contactScore,
    localReasons: candidate.contactReasons
  };
}

function chooseRealisticContacts(candidates: SelectedContact[], topN: number, reason: string): SelectedContact[] {
  return finalizeSelectedContacts([], candidates, topN).map((candidate) => ({
    ...candidate,
    contactReasons: [...candidate.contactReasons, reason].slice(0, 5)
  }));
}

function finalizeSelectedContacts(
  selected: SelectedContact[],
  candidates: SelectedContact[],
  topN: number
): SelectedContact[] {
  const hasRealisticAlternatives = candidates.some(isRealisticFirstContact);
  const selectedIds = new Set<string>();
  const finalized: SelectedContact[] = [];

  for (const contact of selected) {
    if (!contact.inputId || selectedIds.has(contact.inputId)) {
      continue;
    }
    if (hasRealisticAlternatives && !isRealisticFirstContact(contact)) {
      continue;
    }

    finalized.push(contact);
    selectedIds.add(contact.inputId);
    if (finalized.length >= topN) {
      break;
    }
  }

  const fillCandidates = [
    ...candidates.filter((candidate) => isRealisticFirstContact(candidate)),
    ...candidates
  ];
  for (const candidate of fillCandidates) {
    if (!candidate.inputId || selectedIds.has(candidate.inputId)) {
      continue;
    }

    finalized.push(candidate);
    selectedIds.add(candidate.inputId);
    if (finalized.length >= topN) {
      break;
    }
  }

  return finalized.slice(0, topN).map((contact, index) => ({
    ...contact,
    contactPriority: index + 1
  }));
}

function isRealisticFirstContact(person: NormalizedPerson): boolean {
  return !isUnrealisticFirstContact(person) && !isSupportRole(person);
}

function isUnrealisticFirstContact(person: NormalizedPerson): boolean {
  const text = normalizePersonText(person);
  return /\bceo\b|\bchief executive\b|\bfounder\b|\bco-founder\b|\bpresident\b|\bboard\b|\bchairman\b|\bchairwoman\b/.test(text);
}

function isSupportRole(person: NormalizedPerson): boolean {
  return /\bassistant\b|\bexecutive business partner\b|\bchief of staff\b|\badmin\b|\bcoordinator\b/.test(normalizePersonText(person));
}

function normalizePersonText(person: NormalizedPerson): string {
  return normalizeForCompare(
    [
      person.fullName,
      person.title,
      person.department,
      person.seniority,
      person.valueTier,
      person.influence,
      ...(person.teams ?? [])
    ].join(" ")
  );
}

async function selectContactTargets(
  candidates: SelectedContact[],
  context: ContactContext,
  researchReport: ResearchReport,
  options: ContactAgentOptions
): Promise<SelectedContact[]> {
  const apiKey = options.openaiApiKey ?? getEnvValue("OPENAI_API_KEY");
  const topN = options.topN ?? 3;
  if (!apiKey) {
    return chooseRealisticContacts(candidates, topN, "Fallback selection because OPENAI_API_KEY is missing.");
  }

  try {
    const model = options.model ?? getEnvValue("OPENAI_MODEL") ?? DEFAULT_SELECTOR_MODEL;
      const response = await callOpenAIJson<{
      selectedContacts?: Array<{
        inputId?: string;
        fullName?: string;
        contactScore?: number;
        contactReasons?: string[];
        evidenceUrls?: string[];
      }>;
    }>({
      apiKey,
      model,
      fetchImpl: options.fetchImpl,
      payload: {
        task:
          "Select the top people to contact before any email lookup. Choose exactly the best contacts from the provided candidates using the research evidence.",
        contactContext: context,
        researchReport,
        topN,
        candidates: candidates.map(toResearchCandidate),
        instructions:
          "Return JSON with selectedContacts only. Pick people likely to own or influence buying decisions based on current public research, product fit, org role, authority, and accessibility. Prefer reachable operators such as VP, Head, Director, senior manager, or principal leads for platform engineering, infrastructure, DevEx, cloud, web, or frontend systems. Do not pick CEOs, founders, presidents, board members, assistants, chiefs of staff, or generic team roster entries when realistic operator contacts exist. Each selected contact must include inputId, fullName, contactScore, 1-3 contactReasons, and evidenceUrls when a source supports the pick."
      }
    });

    const selectedIds = new Set<string>();
    const selected = (response.selectedContacts ?? [])
      .map((selection) => {
        const matched = candidates.find(
          (candidate) =>
            candidate.inputId === selection.inputId ||
            normalizeForCompare(candidate.fullName) === normalizeForCompare(selection.fullName ?? "")
        );
        if (!matched || !matched.inputId || selectedIds.has(matched.inputId)) {
          return undefined;
        }

        selectedIds.add(matched.inputId);
        return {
          ...matched,
          contactScore: clampScore(selection.contactScore ?? matched.contactScore),
          contactReasons:
            sanitizeStringArray(selection.contactReasons).length > 0
              ? sanitizeStringArray(selection.contactReasons)
              : matched.contactReasons,
          evidenceUrls: sanitizeStringArray(selection.evidenceUrls)
        };
      })
      .filter((contact): contact is SelectedContact => Boolean(contact))
      .slice(0, topN);

    if (selected.length > 0) {
      return finalizeSelectedContacts(selected, candidates, topN);
    }
  } catch {
    // Fall back to local ranking. The email lookup can still proceed.
  }

  return chooseRealisticContacts(candidates, topN, "Fallback selection after selector model failed.");
}

async function callOpenAIJson<T>(input: {
  apiKey: string;
  model: string;
  payload: unknown;
  fetchImpl?: typeof fetch;
  useWebSearch?: boolean;
  maxOutputTokens?: number;
}): Promise<T> {
  const requestBody: UnknownRecord = {
    model: input.model,
    max_output_tokens: input.maxOutputTokens ?? 2200,
    input: [
      {
        role: "system",
        content:
          "You are a careful GTM research and contact selection agent. Use tools when provided, avoid private or invented facts, and return only valid JSON."
      },
      {
        role: "user",
        content: JSON.stringify(input.payload)
      }
    ]
  };

  if (input.useWebSearch) {
    requestBody["tools"] = [{ type: "web_search" }];
    requestBody["tool_choice"] = "auto";
  }

  if (input.model.startsWith("gpt-5")) {
    requestBody["reasoning"] = { effort: input.useWebSearch ? "low" : "minimal" };
    requestBody["text"] = { verbosity: "low" };
  }

  const response = await (input.fetchImpl ?? fetch)(OPENAI_RESPONSES_URL, {
    method: "POST",
    headers: {
      authorization: `Bearer ${input.apiKey}`,
      "content-type": "application/json"
    },
    body: JSON.stringify(requestBody)
  });

  const text = await response.text();
  const json = text ? (JSON.parse(text) as OpenAIResponse) : {};
  if (!response.ok) {
    throw new Error(`OpenAI API error ${response.status}: ${json.error?.message ?? response.statusText}`);
  }

  return parseJsonObject(extractOpenAIText(json)) as T;
}

async function callHunterEmailFinder(
  person: NormalizedPerson,
  apiKey: string,
  fetchImpl: typeof fetch = fetch
): Promise<HunterEmailFinderResponse> {
  const params = new URLSearchParams({
    domain: person.companyDomain ?? "",
    first_name: person.firstName ?? "",
    last_name: person.lastName ?? "",
    api_key: apiKey
  });

  const response = await fetchImpl(`${HUNTER_EMAIL_FINDER_URL}?${params.toString()}`, {
    method: "GET",
    headers: {
      accept: "application/json"
    }
  });

  const text = await response.text();
  const json = text ? (JSON.parse(text) as HunterEmailFinderResponse) : {};
  if (!response.ok) {
    const detail = json.errors?.map((error) => error.details || error.id).filter(Boolean).join("; ");
    throw new Error(`Hunter API error ${response.status}: ${detail || response.statusText}`);
  }

  return json;
}

function normalizePerson(
  rawPerson: unknown,
  index: number,
  defaultCompanyName?: string,
  defaultCompanyDomain?: string
): NormalizedPerson | undefined {
  const person = asRecord(rawPerson);
  if (!person) {
    return undefined;
  }

  const company = asRecord(person.company) ?? asRecord(person.organization);
  const fullName = readString(person, ["fullName", "name", "personName"]);
  const firstName = readString(person, ["firstName", "first_name"]);
  const lastName = readString(person, ["lastName", "last_name"]);
  const composedName = [firstName, lastName].filter(Boolean).join(" ");
  const normalizedName = fullName || composedName;
  const companyName =
    readString(person, ["companyName", "organizationName", "organization"]) ||
    readString(company, ["name", "companyName", "organizationName"]) ||
    defaultCompanyName;
  const companyDomain =
    normalizeCompanyDomainValue(readString(person, ["companyDomain", "domain", "organizationDomain"])) ||
    normalizeCompanyDomainValue(readString(company, ["domain", "companyDomain", "primaryDomain", "website"])) ||
    defaultCompanyDomain;

  if (!normalizedName || !companyName) {
    return undefined;
  }

  const nameParts = splitName(normalizedName);
  return {
    inputId: readString(person, ["inputId", "id", "personId"]) ?? String(index),
    firstName: firstName ?? nameParts.firstName,
    lastName: lastName ?? nameParts.lastName,
    fullName: normalizedName,
    title: readString(person, ["title", "role", "jobTitle"]) ?? undefined,
    department: readString(person, ["department"]),
    companyName,
    companyDomain,
    linkedinUrl: readString(person, ["linkedinUrl", "linkedin_url", "linkedInUrl"]),
    valueTier: readString(person, ["valueTier"]),
    influence: readString(person, ["influence"]),
    seniority: readString(person, ["seniority"]),
    seniorityRank: readNumber(person, ["seniorityRank"]),
    reportSpan: readNumber(person, ["reportSpan"]),
    directReports: readNumber(person, ["directReports"]),
    reportsTo: readString(person, ["reportsTo"]) ?? null,
    level: readNumber(person, ["level"]) ?? null,
    teams: Array.isArray(person.teams) ? person.teams.filter((team): team is string => typeof team === "string") : undefined
  };
}

function scoreCandidate(person: NormalizedPerson, context: ContactContext): { score: number; reasons: string[] } {
  const text = normalizePersonText(person);
  const productText = normalizeForCompare([context.productCategory, context.competitor].filter(Boolean).join(" "));
  const reasons: string[] = [];
  let score = 0;

  if (person.valueTier === "executive") {
    score += 14;
    reasons.push("Executive approver signal, but usually not the first contact.");
  } else if (person.valueTier === "senior-leader") {
    score += 28;
    reasons.push("Senior leader with likely buying influence.");
  } else if (person.valueTier === "leader") {
    score += 20;
    reasons.push("Leader-level role with likely team influence.");
  }

  if (person.influence === "very-high") {
    score += isUnrealisticFirstContact(person) ? 4 : 20;
    reasons.push("Very high influence in source context.");
  } else if (person.influence === "high") {
    score += 14;
    reasons.push("High influence in source context.");
  } else if (person.influence === "low") {
    score += 4;
  }

  if (typeof person.reportSpan === "number" && person.reportSpan > 0) {
    score += isUnrealisticFirstContact(person) ? Math.min(4, Math.ceil(person.reportSpan / 200)) : Math.min(18, Math.ceil(person.reportSpan / 50));
    reasons.push(`Reported downstream span of ${person.reportSpan}.`);
  }

  if (/\bvp\b|\bhead\b|\bdirector\b|\bmanaging director\b|\bsenior manager\b|\bprincipal\b/.test(text)) {
    score += 18;
    reasons.push("Title suggests reachable operator authority.");
  }

  if (/\bchief\b|\bcto\b|\bcio\b|\bciso\b|\bcpo\b|\bcfo\b|\bcoo\b/.test(text)) {
    score += 6;
    reasons.push("C-suite signal; better as approver than first outreach.");
  }

  if (isUnrealisticFirstContact(person)) {
    score -= 50;
    reasons.push("Too senior or indirect for realistic first outreach.");
  }

  if (/\bassistant\b|\bexecutive business partner\b|\bchief of staff\b/.test(text)) {
    score -= 45;
    reasons.push("Operational support role; deprioritized for first outreach.");
  }

  if (productText.includes("frontend") || productText.includes("deployment") || productText.includes("hosting")) {
    if (/\bengineering\b|\bsoftware\b|\bproduct\b|\binfrastructure\b|\bdeveloper\b/.test(text)) {
      score += 18;
      reasons.push("Relevant to frontend/deployment buying context.");
    }
    if (/\bfinance\b|\bpolicy\b|\blegal\b|\bcompliance\b/.test(text)) {
      score -= 5;
    }
  }

  if (person.title === undefined && person.valueTier === "team-member") {
    score -= 25;
    reasons.push("Team-roster entry lacks title or authority signal.");
  }

  return {
    score: clampScore(score),
    reasons: reasons.slice(0, 4)
  };
}

function extractHunterWorkEmail(
  response: HunterEmailFinderResponse,
  expectedDomain?: string
): { email: string; notes: string[] } | undefined {
  const data = response.data;
  const email = data?.email?.trim().toLowerCase();
  if (!email || !isEmail(email)) {
    return undefined;
  }

  const emailDomain = email.split("@")[1];
  if (expectedDomain && !domainsMatch(emailDomain, expectedDomain)) {
    return undefined;
  }

  const notes = [
    `Hunter score: ${typeof data?.score === "number" ? data.score : "unknown"}.`,
    `Hunter verification status: ${data?.verification?.status ?? "unknown"}.`
  ];
  if (expectedDomain) {
    notes.push(`Email domain matched ${expectedDomain}.`);
  }

  return { email, notes };
}

function confidenceFromHunter(response: HunterEmailFinderResponse): Confidence {
  const score = response.data?.score ?? 0;
  const verificationStatus = normalizeForCompare(response.data?.verification?.status ?? "");
  if (score >= 90 && verificationStatus === "valid") {
    return "high";
  }
  if (score >= 70 || verificationStatus === "valid") {
    return "medium";
  }
  return "low";
}

function createBaseResult(person: NormalizedPerson): Omit<EmailFinderResultItem, "status" | "confidence" | "matchMethod" | "notes"> {
  return {
    inputId: person.inputId,
    fullName: person.fullName,
    companyName: person.companyName,
    companyDomain: person.companyDomain,
    provider: "hunter"
  };
}

function summarizeResults(results: EmailFinderResultItem[]): EmailFinderResult["summary"] {
  return {
    total: results.length,
    found: results.filter((result) => result.status === "found").length,
    notFound: results.filter((result) => result.status === "not_found").length,
    ambiguous: results.filter((result) => result.status === "ambiguous").length,
    errors: results.filter((result) => result.status === "error").length
  };
}

function extractOpenAIText(response: OpenAIResponse): string {
  if (response.output_text) {
    return response.output_text;
  }

  const text = response.output
    ?.flatMap((item) => item.content ?? [])
    .map((content) => content.text ?? "")
    .join("")
    .trim();

  if (!text) {
    throw new Error("OpenAI response did not contain text output.");
  }

  return text;
}

function parseJsonObject(text: string): unknown {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const start = trimmed.indexOf("{");
    const end = trimmed.lastIndexOf("}");
    if (start === -1 || end === -1 || end <= start) {
      throw new Error("Model response was not valid JSON.");
    }

    return JSON.parse(trimmed.slice(start, end + 1));
  }
}

function isHunterNotFoundError(error: unknown): boolean {
  const message = error instanceof Error ? error.message.toLowerCase() : "";
  return message.includes("404") || message.includes("not found");
}

function readString(record: UnknownRecord | undefined, keys: string[]): string | undefined {
  if (!record) {
    return undefined;
  }

  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }

  return undefined;
}

function readNumber(record: UnknownRecord | undefined, keys: string[]): number | undefined {
  if (!record) {
    return undefined;
  }

  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }

  return undefined;
}

function sanitizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean);
}

function sanitizeString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizePeopleApiInput(input: unknown, companyDomain: string): unknown {
  const domain = normalizeCompanyDomainHint(companyDomain);
  const companyName = titleCaseDomainLabel(companyDomain);
  const root = asRecord(input);
  const data = asRecord(root?.data);
  const people =
    getPeopleArray(input) ??
    (Array.isArray(root?.results) ? root.results : undefined) ??
    (Array.isArray(root?.contacts) ? root.contacts : undefined) ??
    (Array.isArray(data?.results) ? data.results : undefined) ??
    (Array.isArray(data?.contacts) ? data.contacts : undefined);

  if (Array.isArray(input)) {
    return {
      company: { name: companyName, domain },
      people: input
    };
  }

  if (data && Array.isArray(people)) {
    const apiCompany = asRecord(data.company);
    return {
      ...data,
      company: {
        ...(apiCompany ?? {}),
        name: readString(apiCompany, ["name", "companyName", "organizationName"]) ?? companyName,
        domain: normalizeCompanyDomainValue(readString(apiCompany, ["domain", "companyDomain", "primaryDomain", "website"])) ?? domain
      },
      people
    };
  }

  if (root && Array.isArray(people)) {
    const apiCompany = asRecord(root.company);
    return {
      ...root,
      company: {
        ...(apiCompany ?? {}),
        name: readString(apiCompany, ["name", "companyName", "organizationName"]) ?? companyName,
        domain: normalizeCompanyDomainValue(readString(apiCompany, ["domain", "companyDomain", "primaryDomain", "website"])) ?? domain
      },
      people
    };
  }

  return input;
}

function asRecord(value: unknown): UnknownRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as UnknownRecord) : undefined;
}

function splitName(fullName: string): { firstName?: string; lastName?: string } {
  const parts = fullName.split(/\s+/).filter(Boolean);
  return {
    firstName: parts[0],
    lastName: parts.length > 1 ? parts.slice(1).join(" ") : undefined
  };
}

function normalizeDomain(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }

  return value
    .toLowerCase()
    .replace(/^https?:\/\//, "")
    .replace(/^www\./, "")
    .replace(/^@/, "")
    .split("/")[0]
    .trim();
}

function normalizeCompanyDomainValue(value: string | undefined): string | undefined {
  const normalized = normalizeDomain(value);
  if (!normalized) {
    return undefined;
  }

  return normalized.includes(".") ? normalized : `${normalized}.com`;
}

function normalizeCompanyDomainHint(value: string): string {
  return normalizeCompanyDomainValue(value) ?? value.trim().toLowerCase();
}

function titleCaseDomainLabel(value: string): string {
  const label = (normalizeDomain(value) ?? value)
    .split(".")[0]
    .replace(/[-_]+/g, " ")
    .trim();
  return label
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1).toLowerCase()}`)
    .join(" ");
}

function domainsMatch(actual: string | undefined, expected: string | undefined): boolean {
  const normalizedActual = normalizeDomain(actual);
  const normalizedExpected = normalizeDomain(expected);
  return Boolean(
    normalizedActual &&
      normalizedExpected &&
      (normalizedActual === normalizedExpected || normalizedActual.endsWith(`.${normalizedExpected}`))
  );
}

function isEmail(value: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

function compareOptionalNumber(a: number | undefined, b: number | undefined): number {
  if (a === undefined && b === undefined) {
    return 0;
  }
  if (a === undefined) {
    return 1;
  }
  if (b === undefined) {
    return -1;
  }
  return a - b;
}

function clampScore(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(100, Math.round(value)));
}

function normalizeForCompare(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function getEnvValue(name: string): string | undefined {
  const processValue = process.env[name];
  if (processValue) {
    return processValue;
  }

  try {
    const envFile = readFileSync(resolve(".env"), "utf8");
    for (const line of envFile.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) {
        continue;
      }

      const [key, ...valueParts] = trimmed.split("=");
      if (key === name) {
        return valueParts.join("=").replace(/^["']|["']$/g, "");
      }
    }
  } catch {
    return undefined;
  }

  return undefined;
}

function parseCliArgs(argv: string[]): {
  inputPath: string;
  outputPath: string;
  domain?: string;
  context: ContactContext;
  topN?: number;
  candidateLimit?: number;
} {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      continue;
    }

    args.set(key.slice(2), value);
    index += 1;
  }

  const inputPath = args.get("input") ?? "coinbase-org-context.json";
  const domain = args.get("domain");
  const topN = args.has("top") ? Number(args.get("top")) : undefined;
  const candidateLimit = args.has("candidateLimit") ? Number(args.get("candidateLimit")) : undefined;

  return {
    inputPath,
    outputPath: args.get("output") ?? (domain ? `${domain}-selected-contacts-with-emails.json` : getDefaultOutputPath(inputPath)),
    domain,
    topN: Number.isFinite(topN) ? topN : undefined,
    candidateLimit: Number.isFinite(candidateLimit) ? candidateLimit : undefined,
    context: {
      sellerCompany: args.get("sellerCompany"),
      targetCompany: args.get("targetCompany"),
      productCategory: args.get("productCategory"),
      competitor: args.get("competitor")
    }
  };
}

function getPeopleArray(input: unknown): unknown[] | undefined {
  const root = asRecord(input);
  if (Array.isArray(root?.people)) {
    return root.people;
  }

  const data = asRecord(root?.data);
  if (Array.isArray(data?.people)) {
    return data.people;
  }

  return Array.isArray(input) ? input : undefined;
}

function getCompanyRecord(root: UnknownRecord | undefined, selectedContacts: SelectedContact[]): UnknownRecord {
  const data = asRecord(root?.data);
  const company = asRecord(root?.company) ?? asRecord(data?.company);
  if (company) {
    return structuredCloneJson(company) as UnknownRecord;
  }

  const firstContact = selectedContacts[0];
  return {
    name: firstContact?.companyName ?? "Unknown company",
    domain: firstContact?.companyDomain
  };
}

function createPersonRecordFromContact(contact: SelectedContact): UnknownRecord {
  return {
    inputId: contact.inputId,
    name: contact.fullName,
    firstName: contact.firstName,
    lastName: contact.lastName,
    title: contact.title,
    department: contact.department,
    companyName: contact.companyName,
    companyDomain: contact.companyDomain,
    linkedinUrl: contact.linkedinUrl
  };
}

function findInputPersonIndex(people: unknown[], inputId: string | undefined): number {
  const numericIndex = inputId !== undefined ? Number(inputId) : Number.NaN;
  if (Number.isInteger(numericIndex) && numericIndex >= 0 && numericIndex < people.length) {
    return numericIndex;
  }

  return people.findIndex((person) => {
    const record = asRecord(person);
    if (!record) {
      return false;
    }

    const id = readString(record, ["inputId", "id", "personId"]);
    return id !== undefined && id === inputId;
  });
}

function structuredCloneJson(input: unknown): unknown {
  return JSON.parse(JSON.stringify(input)) as unknown;
}

function getDefaultOutputPath(inputPath: string): string {
  const directory = dirname(inputPath);
  const extension = extname(inputPath) || ".json";
  const name = basename(inputPath, extname(inputPath));
  return join(directory, `${name}-with-emails${extension}`);
}

const isCliRun = process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1]);

if (isCliRun) {
  const { inputPath, outputPath, domain, context, topN, candidateLimit } = parseCliArgs(process.argv.slice(2));
  const input = domain ? await loadPeopleInputFromApi(domain) : loadPeopleInput(inputPath);
  const inputRoot = asRecord(input);
  const company = asRecord(inputRoot?.company) ?? asRecord(asRecord(inputRoot?.data)?.company);
  const result = await runContactAgent(
    input,
    {
      ...context,
      targetCompany: context.targetCompany ?? readString(company, ["name"])
    },
    {
      topN,
      candidateLimit
    }
  );
  writeFileSync(resolve(outputPath), `${JSON.stringify(result.enrichedJson, null, 2)}\n`);
  console.log(
    JSON.stringify(
      {
        outputPath,
        selectedContacts: result.selectedContacts.map((contact) => ({
          name: contact.fullName,
          title: contact.title,
          contactScore: contact.contactScore
        })),
        emailSummary: result.emailLookup.summary,
        trace: result.trace
      },
      null,
      2
    )
  );
}
