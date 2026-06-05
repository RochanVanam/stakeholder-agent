import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type PersonId = string;

export interface Person {
  id: PersonId;
  name: string;
  title: string;
  department: string;
  level: number;
  reportsTo: PersonId | null;
}

export interface OrgChart {
  rootId?: PersonId;
  people?: Person[];
  peopleMap?: Record<PersonId, Person>;
  childrenMap?: Record<PersonId, PersonId[]>;
}

export interface ProductContext {
  sellerCompany: string;
  targetCompany: string;
  productCategory: string;
  competitor?: string;
}

export interface ScoreResult {
  score: number;
  authorityScore: number;
  relevanceScore: number;
  matchedKeywords: string[];
  reasons: string[];
}

export interface Stakeholder extends Person {
  score: number;
  matchedKeywords: string[];
  reasons: string[];
}

export interface ResearchPlan {
  researchQuestions: string[];
  searchQueries: string[];
  roleHypotheses: string[];
  evidenceNeeded: string[];
}

export interface ResearchSource {
  title: string;
  url: string;
  relevance: string;
}

export interface ResearchReport {
  companySignals: string[];
  productFitSignals: string[];
  competitorSignals: string[];
  stakeholderSignals: Array<{
    personId?: PersonId;
    roleOrTeam: string;
    signal: string;
    sourceUrls: string[];
  }>;
  recommendedAngles: string[];
  sources: ResearchSource[];
}

export interface OutreachDraft {
  personId: PersonId;
  name: string;
  subject: string;
  email: string;
  rationale: string;
}

export interface AgentTraceStep {
  step: string;
  why: string;
  outputSummary: string;
}

export interface StakeholderAgentResult {
  productContext: ProductContext;
  agentMode: "research-agent";
  model: string;
  researchPlan: ResearchPlan;
  researchReport: ResearchReport;
  primaryBuyers: Stakeholder[];
  influencers: Stakeholder[];
  executiveApprovers: Stakeholder[];
  pathToDecisionMaker: Stakeholder[];
  outreachDrafts: OutreachDraft[];
  summary: string;
  reasoning: {
    scoringPriorities: string[];
    primaryBuyerCriteria: string;
    influencerCriteria: string;
    executiveApproverCriteria: string;
  };
  trace: AgentTraceStep[];
}

export interface StakeholderAgentOptions {
  model?: string;
  apiKey?: string;
}

type LoadableOrgChart = OrgChart | string | URL;

interface OpenAIResponse {
  status?: string;
  incomplete_details?: {
    reason?: string;
  };
  output_text?: string;
  output?: Array<{
    type?: string;
    content?: Array<{
      text?: string;
    }>;
  }>;
  error?: {
    message?: string;
  };
}

interface CliArgs {
  orgPath: string;
  productContext: ProductContext;
  options: StakeholderAgentOptions;
}

interface OpenAICallInput {
  options: StakeholderAgentOptions;
  task: string;
  payload: unknown;
  systemPrompt?: string;
  useWebSearch?: boolean;
  maxOutputTokens?: number;
}

const DEFAULT_GPT_MODEL = "gpt-5-nano";

const STAKEHOLDER_AGENT_SYSTEM_PROMPT = [
  "You are the Stakeholder Agent in a GTM Agent Broker.",
  "Work like a human GTM researcher: form hypotheses, research the account, map evidence to roles, then decide who to contact.",
  "Return only valid JSON. Do not wrap JSON in markdown.",
  "Every selected stakeholder must come from the provided org chart.",
  "Keep each selected stakeholder's id, name, title, department, level, and reportsTo exactly as provided.",
  "Scores must be integers from 0 to 100.",
  "Each selected stakeholder must include 1-2 non-empty reasons under 16 words each.",
  "For productCategory = frontend hosting/deployment, prioritize platform engineering, web infrastructure, developer experience, frontend infrastructure, hosting, deployment, site reliability, engineering tooling, and cloud infrastructure.",
  "If research evidence conflicts with the org chart, use the org chart for identity and reporting structure, and research for account-specific rationale."
].join(" ");

export function loadOrgChart(input: LoadableOrgChart = "example.json"): OrgChart {
  if (typeof input === "object" && !(input instanceof URL)) {
    return normalizeOrgChart(input);
  }

  const path = input instanceof URL ? fileURLToPath(input) : resolve(input);
  const raw = readFileSync(path, "utf8");
  return normalizeOrgChart(JSON.parse(raw) as OrgChart);
}

export function flattenOrgTree(orgChart: OrgChart): Person[] {
  const normalized = normalizeOrgChart(orgChart);
  const peopleMap = normalized.peopleMap ?? {};
  const childrenMap = normalized.childrenMap ?? buildChildrenMap(Object.values(peopleMap));
  const rootId = normalized.rootId ?? findRootId(Object.values(peopleMap));
  const visited = new Set<PersonId>();
  const flattened: Person[] = [];

  const visit = (id: PersonId): void => {
    const person = peopleMap[id];
    if (!person || visited.has(id)) {
      return;
    }

    visited.add(id);
    flattened.push(person);

    for (const childId of childrenMap[id] ?? []) {
      visit(childId);
    }
  };

  if (rootId) {
    visit(rootId);
  }

  for (const person of Object.values(peopleMap).sort(byOrgPosition)) {
    if (!visited.has(person.id)) {
      visit(person.id);
    }
  }

  return flattened;
}

export async function scorePerson(
  person: Person,
  context: ProductContext,
  options: StakeholderAgentOptions = {}
): Promise<ScoreResult> {
  return callOpenAIJson<ScoreResult>({
    options,
    task: "Score this single person for the buying committee.",
    payload: {
      productContext: context,
      person,
      instructions:
        "Return exactly these fields: score, authorityScore, relevanceScore, matchedKeywords, reasons. Reasons must contain 1-2 non-empty short strings."
    }
  });
}

export async function planResearch(
  orgChart: OrgChart,
  context: ProductContext,
  options: StakeholderAgentOptions = {}
): Promise<ResearchPlan> {
  const normalized = normalizeOrgChart(orgChart);
  const people = flattenOrgTree(normalized);
  const plan = await callOpenAIJson<Partial<ResearchPlan>>({
    options,
    task: "Create a concise account research plan before choosing stakeholders.",
    payload: {
      productContext: context,
      orgChartSummary: summarizeOrgForPrompt(people),
      instructions:
        "Return researchQuestions, searchQueries, roleHypotheses, and evidenceNeeded. Include queries about target company engineering priorities, web/frontend/developer platform initiatives, infrastructure strategy, and competitor context."
    },
    maxOutputTokens: 1800
  });

  return sanitizeResearchPlan(plan, context);
}

export async function conductCompanyResearch(
  orgChart: OrgChart,
  context: ProductContext,
  researchPlan: ResearchPlan,
  options: StakeholderAgentOptions = {}
): Promise<ResearchReport> {
  const normalized = normalizeOrgChart(orgChart);
  const people = flattenOrgTree(normalized);
  const report = await callOpenAIJson<Partial<ResearchReport>>({
    options,
    task: "Research the account like a human sales researcher, then summarize evidence for stakeholder selection.",
    payload: {
      productContext: context,
      researchPlan,
      orgChartPeople: summarizeOrgForPrompt(people),
      instructions:
        "Use web research to find public evidence about the target company's engineering priorities, product launches, frontend/web properties, developer platform needs, infrastructure strategy, reliability concerns, and competitor context. Return non-empty companySignals, productFitSignals, competitorSignals, stakeholderSignals, recommendedAngles, and sources. Each stakeholderSignal must include a non-empty roleOrTeam, signal, and sourceUrls. Include URLs in sources and sourceUrls."
    },
    useWebSearch: true,
    maxOutputTokens: 5000
  });

  return sanitizeResearchReport(report);
}

export async function classifyStakeholders(
  orgChart: OrgChart,
  context: ProductContext,
  options: StakeholderAgentOptions = {},
  researchReport?: ResearchReport
): Promise<StakeholderAgentResult> {
  const normalized = normalizeOrgChart(orgChart);
  const model = options.model ?? getEnvValue("OPENAI_MODEL") ?? DEFAULT_GPT_MODEL;
  const people = flattenOrgTree(normalized);
  const effectiveResearchReport =
    researchReport ?? (await conductCompanyResearch(normalized, context, await planResearch(normalized, context, options), options));
  const result = await callOpenAIJson<Partial<StakeholderAgentResult>>({
    options: { ...options, model },
    task: "Use the research report and org chart to select the stakeholder buying committee.",
    payload: {
      productContext: context,
      researchReport: effectiveResearchReport,
      orgChart: {
        rootId: normalized.rootId,
        people,
        childrenMap: normalized.childrenMap
      },
      instructions:
        "Return exactly these top-level fields: productContext, agentMode, model, primaryBuyers, influencers, executiveApprovers, pathToDecisionMaker, summary, reasoning. " +
        "primaryBuyers must contain 3-5 people who own evaluation and buying criteria. " +
        "influencers must contain 3-6 people who validate technical fit, migration risk, developer experience, and incumbent replacement concerns. " +
        "executiveApprovers must contain 2-4 senior leaders likely to approve budget, strategic platform risk, security, legal, or procurement. " +
        "pathToDecisionMaker must be a valid reportsTo chain from the strongest primary buyer upward. " +
        "Each stakeholder must include id, name, title, department, level, reportsTo, score, matchedKeywords, and reasons. Reasons should reference research evidence when possible. " +
        "Do not include schema examples, outputContract, stakeholderShape, or any extra top-level fields."
    },
    maxOutputTokens: 5000
  });

  const placeholderPlan = sanitizeResearchPlan({}, context);
  return sanitizeStakeholderAgentResult(result, context, model, normalized, placeholderPlan, effectiveResearchReport, []);
}

export async function draftOutreach(
  stakeholders: Stakeholder[],
  context: ProductContext,
  researchReport: ResearchReport,
  options: StakeholderAgentOptions = {}
): Promise<OutreachDraft[]> {
  const drafts = await callOpenAIJson<unknown>({
    systemPrompt:
      "You are a concise enterprise sales researcher. Return only valid JSON with first-touch outreach drafts. Do not wrap JSON in markdown.",
    options,
    task: "Draft concise first-touch outreach emails for the selected stakeholders.",
    payload: {
      productContext: context,
      stakeholders: stakeholders.slice(0, 3),
      researchReport,
      instructions:
        "Return exactly one top-level field named outreachDrafts. Draft one email body for each provided stakeholder. Each draft must include the exact personId, exact name, subject, email, and rationale. The email field must be the message body, not an email address. Do not invent recipient email addresses, private facts, or personal details. Keep emails short, specific, and grounded in the research report."
    },
    maxOutputTokens: 3500
  });

  return sanitizeOutreachDrafts(extractOutreachDraftArray(drafts), stakeholders);
}

export function getPathToRoot(personId: PersonId, orgChart: OrgChart): Person[] {
  const normalized = normalizeOrgChart(orgChart);
  const peopleMap = normalized.peopleMap ?? {};
  const path: Person[] = [];
  const visited = new Set<PersonId>();
  let currentId: PersonId | null = personId;

  while (currentId) {
    if (visited.has(currentId)) {
      throw new Error(`Cycle detected while walking management chain at ${currentId}`);
    }

    visited.add(currentId);
    const person = peopleMap[currentId];
    if (!person) {
      break;
    }

    path.push(person);
    currentId = person.reportsTo;
  }

  return path;
}

export async function runStakeholderAgent(input: {
  orgChart?: LoadableOrgChart;
  productContext: ProductContext;
  options?: StakeholderAgentOptions;
}): Promise<StakeholderAgentResult> {
  const orgChart = loadOrgChart(input.orgChart ?? "example.json");
  const model = input.options?.model ?? getEnvValue("OPENAI_MODEL") ?? DEFAULT_GPT_MODEL;
  const trace: AgentTraceStep[] = [];

  const researchPlan = await planResearch(orgChart, input.productContext, input.options);
  trace.push({
    step: "planResearch",
    why: "A human seller starts by deciding what evidence would change the contact strategy.",
    outputSummary: `${researchPlan.searchQueries.length} search queries and ${researchPlan.roleHypotheses.length} role hypotheses.`
  });

  const researchReport = await conductCompanyResearch(orgChart, input.productContext, researchPlan, input.options);
  trace.push({
    step: "conductCompanyResearch",
    why: "Public account evidence grounds the buying committee in current priorities, not just titles.",
    outputSummary: `${researchReport.sources.length} sources and ${researchReport.recommendedAngles.length} recommended angles.`
  });

  const classified = await classifyStakeholders(orgChart, input.productContext, input.options, researchReport);
  trace.push({
    step: "classifyStakeholders",
    why: "The agent maps researched account signals back to the org chart and management paths.",
    outputSummary: `${classified.primaryBuyers.length} primary buyers, ${classified.influencers.length} influencers, ${classified.executiveApprovers.length} approvers.`
  });

  const outreachDrafts = await draftOutreach(classified.primaryBuyers, input.productContext, researchReport, input.options);
  trace.push({
    step: "draftOutreach",
    why: "A human researcher turns the stakeholder decision into a concrete next action.",
    outputSummary: `${outreachDrafts.length} first-touch email drafts.`
  });

  return {
    ...classified,
    productContext: input.productContext,
    agentMode: "research-agent",
    model,
    researchPlan,
    researchReport,
    outreachDrafts,
    trace
  };
}

async function callOpenAIJson<T>(input: OpenAICallInput): Promise<T> {
  const apiKey = input.options.apiKey ?? getEnvValue("OPENAI_API_KEY");
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required. Add it to .env or export it in your shell.");
  }

  const model = input.options.model ?? getEnvValue("OPENAI_MODEL") ?? DEFAULT_GPT_MODEL;
  const requestBody: Record<string, unknown> = {
    model,
    max_output_tokens: input.maxOutputTokens ?? 6000,
    input: [
      {
        role: "system",
        content: input.systemPrompt ?? STAKEHOLDER_AGENT_SYSTEM_PROMPT
      },
      {
        role: "user",
        content: JSON.stringify({
          task: input.task,
          ...input.payload
        })
      }
    ]
  };

  if (input.useWebSearch) {
    requestBody.tools = [{ type: "web_search" }];
    requestBody.tool_choice = "auto";
    requestBody.include = ["web_search_call.action.sources"];
  }

  if (model.startsWith("gpt-5")) {
    requestBody.reasoning = { effort: input.useWebSearch ? "low" : "minimal" };
    requestBody.text = { verbosity: "low" };
  }

  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(requestBody)
  });

  const json = (await response.json()) as OpenAIResponse;
  if (!response.ok) {
    throw new Error(`OpenAI API error: ${json.error?.message ?? response.statusText}`);
  }

  return parseJsonObject(extractResponseText(json)) as T;
}

function normalizeOrgChart(orgChart: OrgChart): OrgChart {
  const people = orgChart.people ?? Object.values(orgChart.peopleMap ?? {});
  const peopleMap = orgChart.peopleMap ?? Object.fromEntries(people.map((person) => [person.id, person]));
  const childrenMap = orgChart.childrenMap ?? buildChildrenMap(people);
  const rootId = orgChart.rootId ?? findRootId(people);

  return {
    rootId,
    people: Object.values(peopleMap).sort(byOrgPosition),
    peopleMap,
    childrenMap
  };
}

function buildChildrenMap(people: Person[]): Record<PersonId, PersonId[]> {
  const childrenMap: Record<PersonId, PersonId[]> = {};

  for (const person of people) {
    childrenMap[person.id] ??= [];
    if (person.reportsTo) {
      childrenMap[person.reportsTo] ??= [];
      childrenMap[person.reportsTo].push(person.id);
    }
  }

  for (const children of Object.values(childrenMap)) {
    children.sort();
  }

  return childrenMap;
}

function findRootId(people: Person[]): PersonId | undefined {
  return people.find((person) => person.reportsTo === null)?.id;
}

function summarizeOrgForPrompt(people: Person[]): Person[] {
  return people.filter((person) => person.level <= 4);
}

function sanitizeStakeholderAgentResult(
  result: Partial<StakeholderAgentResult>,
  context: ProductContext,
  model: string,
  orgChart: OrgChart,
  researchPlan: ResearchPlan,
  researchReport: ResearchReport,
  outreachDrafts: OutreachDraft[]
): StakeholderAgentResult {
  result = unwrapStakeholderAgentResult(result);
  const normalized = normalizeOrgChart(orgChart);
  const peopleMap = normalized.peopleMap ?? {};
  const scoreById = new Map<PersonId, Pick<Stakeholder, "score" | "matchedKeywords" | "reasons">>();
  const returnedStakeholders = [
    ...(result.primaryBuyers ?? []),
    ...(result.influencers ?? []),
    ...(result.executiveApprovers ?? []),
    ...(result.pathToDecisionMaker ?? [])
  ];

  for (const stakeholder of returnedStakeholders) {
    if (!stakeholder?.id) {
      continue;
    }

    const existing = scoreById.get(stakeholder.id);
    const nextScore = clampScore(stakeholder.score ?? 0);
    if (existing && existing.score > nextScore) {
      continue;
    }

    scoreById.set(stakeholder.id, {
      score: stakeholder.score ?? 0,
      matchedKeywords: Array.isArray(stakeholder.matchedKeywords) ? stakeholder.matchedKeywords : [],
      reasons: Array.isArray(stakeholder.reasons) ? stakeholder.reasons : []
    });
  }

  const primaryBuyers = sanitizeStakeholders(result.primaryBuyers, peopleMap, scoreById);
  const influencers = sanitizeStakeholders(result.influencers, peopleMap, scoreById);
  const executiveApprovers = sanitizeStakeholders(result.executiveApprovers, peopleMap, scoreById);
  const pathToDecisionMaker = sanitizePathToDecisionMaker(
    result.pathToDecisionMaker,
    primaryBuyers,
    peopleMap,
    scoreById,
    normalized
  );

  return {
    productContext: context,
    agentMode: "research-agent",
    model,
    researchPlan,
    researchReport,
    primaryBuyers,
    influencers,
    executiveApprovers,
    pathToDecisionMaker,
    outreachDrafts,
    summary: typeof result.summary === "string" ? result.summary : "",
    reasoning: {
      scoringPriorities: sanitizeStringArray(result.reasoning?.scoringPriorities),
      primaryBuyerCriteria: sanitizeString(result.reasoning?.primaryBuyerCriteria),
      influencerCriteria: sanitizeString(result.reasoning?.influencerCriteria),
      executiveApproverCriteria: sanitizeString(result.reasoning?.executiveApproverCriteria)
    },
    trace: []
  };
}

function sanitizePathToDecisionMaker(
  path: Stakeholder[] | undefined,
  primaryBuyers: Stakeholder[],
  peopleMap: Record<PersonId, Person>,
  scoreById: Map<PersonId, Pick<Stakeholder, "score" | "matchedKeywords" | "reasons">>,
  orgChart: OrgChart
): Stakeholder[] {
  const sanitizedPath = sanitizeStakeholders(path, peopleMap, scoreById);
  if (sanitizedPath.length >= 2 && isValidReportsToChain(sanitizedPath)) {
    return sanitizedPath;
  }

  const anchor = primaryBuyers[0];
  if (!anchor) {
    return sanitizedPath;
  }

  return getPathToRoot(anchor.id, orgChart).map((person) => {
    const scoreData = scoreById.get(person.id);
    return {
      ...person,
      score: clampScore(scoreData?.score ?? 0),
      matchedKeywords: sanitizeStringArray(scoreData?.matchedKeywords),
      reasons: sanitizeStringArray(scoreData?.reasons)
    };
  });
}

function isValidReportsToChain(path: Stakeholder[]): boolean {
  return path.every((person, index) => {
    const nextPerson = path[index + 1];
    return !nextPerson || person.reportsTo === nextPerson.id;
  });
}

function unwrapStakeholderAgentResult(result: Partial<StakeholderAgentResult>): Partial<StakeholderAgentResult> {
  const wrapped = result as Partial<StakeholderAgentResult> & {
    result?: Partial<StakeholderAgentResult>;
    stakeholderAgentResult?: Partial<StakeholderAgentResult>;
  };

  return wrapped.result ?? wrapped.stakeholderAgentResult ?? result;
}

function sanitizeStakeholders(
  stakeholders: Stakeholder[] | undefined,
  peopleMap: Record<PersonId, Person>,
  scoreById: Map<PersonId, Pick<Stakeholder, "score" | "matchedKeywords" | "reasons">>
): Stakeholder[] {
  if (!Array.isArray(stakeholders)) {
    return [];
  }

  return stakeholders
    .map((stakeholder) => {
      const person = peopleMap[stakeholder.id];
      if (!person) {
        return undefined;
      }

      const scoreData = scoreById.get(stakeholder.id);
      return {
        ...person,
        score: clampScore(stakeholder.score ?? scoreData?.score ?? 0),
        matchedKeywords: sanitizeStringArray(stakeholder.matchedKeywords ?? scoreData?.matchedKeywords),
        reasons: sanitizeStringArray(stakeholder.reasons ?? scoreData?.reasons)
      };
    })
    .filter((stakeholder): stakeholder is Stakeholder => Boolean(stakeholder));
}

function sanitizeResearchPlan(plan: Partial<ResearchPlan>, context: ProductContext): ResearchPlan {
  return {
    researchQuestions: sanitizeStringArray(plan.researchQuestions).slice(0, 8),
    searchQueries: sanitizeStringArray(plan.searchQueries).slice(0, 8),
    roleHypotheses: sanitizeStringArray(plan.roleHypotheses).slice(0, 8),
    evidenceNeeded:
      sanitizeStringArray(plan.evidenceNeeded).slice(0, 8).length > 0
        ? sanitizeStringArray(plan.evidenceNeeded).slice(0, 8)
        : [
            `${context.targetCompany} engineering priorities`,
            `${context.productCategory} ownership signals`,
            `${context.competitor ?? "incumbent"} replacement risks`
          ]
  };
}

function sanitizeResearchReport(report: Partial<ResearchReport>): ResearchReport {
  const sources = sanitizeResearchSources(report.sources);
  const companySignals = sanitizeStringArray(report.companySignals);
  const productFitSignals = sanitizeStringArray(report.productFitSignals);
  const competitorSignals = sanitizeStringArray(report.competitorSignals);

  return {
    companySignals:
      companySignals.length > 0
        ? companySignals
        : sources.slice(0, 3).map((source) => `Found public account evidence from ${source.title}.`),
    productFitSignals:
      productFitSignals.length > 0
        ? productFitSignals
        : sources.slice(0, 3).map((source) => `Source may inform frontend hosting/deployment fit: ${source.title}.`),
    competitorSignals,
    stakeholderSignals: Array.isArray(report.stakeholderSignals)
      ? report.stakeholderSignals.map((signal) => ({
          personId: typeof signal.personId === "string" ? signal.personId : undefined,
          roleOrTeam: sanitizeString(signal.roleOrTeam),
          signal: sanitizeString(signal.signal),
          sourceUrls: sanitizeStringArray(signal.sourceUrls)
        })).filter((signal) => signal.roleOrTeam || signal.signal || signal.sourceUrls.length > 0)
      : [],
    recommendedAngles: sanitizeStringArray(report.recommendedAngles),
    sources
  };
}

function sanitizeResearchSources(sources: ResearchSource[] | undefined): ResearchSource[] {
  if (!Array.isArray(sources)) {
    return [];
  }

  return sources
    .map((source) => ({
      title: sanitizeString(source.title),
      url: sanitizeString(source.url),
      relevance: sanitizeString(source.relevance)
    }))
    .filter((source) => source.url);
}

function extractOutreachDraftArray(value: unknown): unknown[] | undefined {
  if (Array.isArray(value)) {
    return value;
  }

  if (!value || typeof value !== "object") {
    return undefined;
  }

  const record = value as Record<string, unknown>;
  const candidates = [
    record.outreachDrafts,
    record.drafts,
    record.emails,
    record.outreach,
    record.results,
    record.messages
  ];

  return candidates.find((candidate): candidate is unknown[] => Array.isArray(candidate));
}

function sanitizeOutreachDrafts(drafts: unknown[] | undefined, stakeholders: Stakeholder[]): OutreachDraft[] {
  if (!Array.isArray(drafts)) {
    return [];
  }

  const allowedIds = new Set(stakeholders.map((stakeholder) => stakeholder.id));
  const idByName = new Map(stakeholders.map((stakeholder) => [stakeholder.name, stakeholder.id]));
  return drafts
    .map((draft) => ({
      personId: sanitizeDraftPersonId(draft, idByName),
      name: sanitizeString(readDraftField(draft, ["name", "recipient", "to"])),
      subject: sanitizeString(readDraftField(draft, ["subject", "subjectLine"])),
      email: sanitizeEmailBody(readDraftField(draft, ["email", "body", "emailBody", "message"])),
      rationale: sanitizeString(readDraftField(draft, ["rationale", "why", "reason"]))
    }))
    .filter((draft) => allowedIds.has(draft.personId) && draft.subject && draft.email);
}

function sanitizeDraftPersonId(value: unknown, idByName: Map<string, PersonId>): string {
  const directId = sanitizeString(readDraftField(value, ["personId", "personID", "id", "stakeholderId"]));
  if (directId) {
    return directId;
  }

  const name = sanitizeString(readDraftField(value, ["name", "recipient", "to"]));
  return idByName.get(name) ?? "";
}

function sanitizeEmailBody(value: unknown): string {
  const text = sanitizeString(value);
  if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(text)) {
    return "";
  }

  return text;
}

function readDraftField(value: unknown, fields: string[]): unknown {
  if (!value || typeof value !== "object") {
    return undefined;
  }

  const record = value as Record<string, unknown>;
  return fields.map((field) => record[field]).find((fieldValue) => fieldValue !== undefined);
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
  if (typeof value === "string") {
    return value.trim();
  }

  if (Array.isArray(value)) {
    return sanitizeStringArray(value).join(" ");
  }

  return "";
}

function clampScore(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(100, Math.round(value)));
}

function byOrgPosition(a: Person, b: Person): number {
  return a.level - b.level || a.id.localeCompare(b.id);
}

function extractResponseText(response: OpenAIResponse): string {
  if (response.output_text) {
    return response.output_text;
  }

  const text = response.output
    ?.flatMap((item) => item.content ?? [])
    .map((content) => content.text ?? "")
    .join("")
    .trim();

  if (!text) {
    const status = response.status ? `status=${response.status}` : "status=unknown";
    const reason = response.incomplete_details?.reason ? `, reason=${response.incomplete_details.reason}` : "";
    const outputTypes = response.output?.map((item) => item.type ?? "unknown").join(", ") ?? "none";
    throw new Error(`OpenAI response did not contain text output (${status}${reason}, outputTypes=${outputTypes}).`);
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
      throw new Error("GPT response was not valid JSON.");
    }

    return JSON.parse(trimmed.slice(start, end + 1));
  }
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

function parseCliArgs(argv: string[]): CliArgs {
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

  return {
    orgPath: args.get("org") ?? "example.json",
    productContext: {
      sellerCompany: args.get("sellerCompany") ?? "Vercel",
      targetCompany: args.get("targetCompany") ?? "NVIDIA",
      productCategory: args.get("productCategory") ?? "frontend hosting/deployment",
      competitor: args.get("competitor") ?? "Cloudflare"
    },
    options: {
      model: args.get("model")
    }
  };
}

const isCliRun = process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1]);

if (isCliRun) {
  const { orgPath, productContext, options } = parseCliArgs(process.argv.slice(2));
  const result = await runStakeholderAgent({
    orgChart: orgPath,
    productContext,
    options
  });

  console.log(JSON.stringify(result, null, 2));
}
