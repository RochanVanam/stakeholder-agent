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

export interface StakeholderAgentResult {
  productContext: ProductContext;
  agentMode: "gpt";
  model: string;
  primaryBuyers: Stakeholder[];
  influencers: Stakeholder[];
  executiveApprovers: Stakeholder[];
  pathToDecisionMaker: Stakeholder[];
  summary: string;
  reasoning: {
    scoringPriorities: string[];
    primaryBuyerCriteria: string;
    influencerCriteria: string;
    executiveApproverCriteria: string;
  };
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

const DEFAULT_GPT_MODEL = "gpt-5-nano";

const STAKEHOLDER_AGENT_SYSTEM_PROMPT = [
  "You are the Stakeholder Agent in a GTM Agent Broker.",
  "Your job is to analyze an org chart and product context, then identify the target account's likely buying committee.",
  "You must traverse the reporting structure, score people by relevance, classify stakeholders, and return structured JSON only.",
  "Use GTM judgment, not fixed keyword rules.",
  "Every selected stakeholder must come from the provided org chart.",
  "Keep each selected stakeholder's id, name, title, department, level, and reportsTo exactly as provided.",
  "Scores must be integers from 0 to 100.",
  "Each selected stakeholder must include 1-2 non-empty reasons under 16 words each.",
  "Keep the JSON compact: 3-5 primary buyers, 3-6 influencers, 2-4 executive approvers.",
  "The reasoning object must contain non-empty scoringPriorities, primaryBuyerCriteria, influencerCriteria, and executiveApproverCriteria.",
  "pathToDecisionMaker must include every manager in the reportsTo chain from the strongest primary buyer up to the final executive.",
  "For productCategory = frontend hosting/deployment, prioritize platform engineering, web infrastructure, developer experience, frontend infrastructure, hosting, deployment, site reliability, engineering tooling, and cloud infrastructure.",
  "Return only valid JSON. Do not wrap the result in markdown."
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

export async function classifyStakeholders(
  orgChart: OrgChart,
  context: ProductContext,
  options: StakeholderAgentOptions = {}
): Promise<StakeholderAgentResult> {
  const normalized = normalizeOrgChart(orgChart);
  const model = options.model ?? getEnvValue("OPENAI_MODEL") ?? DEFAULT_GPT_MODEL;
  const people = flattenOrgTree(normalized);

  const result = await callOpenAIJson<Partial<StakeholderAgentResult>>({
    options: { ...options, model },
    task: "Classify the full stakeholder buying committee from this org chart.",
    payload: {
      productContext: context,
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
        "Each stakeholder must include id, name, title, department, level, reportsTo, score, matchedKeywords, and reasons. " +
        "Do not include schema examples, outputContract, stakeholderShape, or any extra top-level fields."
    }
  });

  return sanitizeStakeholderAgentResult(result, context, model, normalized);
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
  return classifyStakeholders(orgChart, input.productContext, input.options);
}

async function callOpenAIJson<T>(input: {
  options: StakeholderAgentOptions;
  task: string;
  payload: unknown;
}): Promise<T> {
  const apiKey = input.options.apiKey ?? getEnvValue("OPENAI_API_KEY");
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required. Add it to .env or export it in your shell.");
  }

  const model = input.options.model ?? getEnvValue("OPENAI_MODEL") ?? DEFAULT_GPT_MODEL;
  const requestBody: Record<string, unknown> = {
    model,
    max_output_tokens: 6000,
    input: [
      {
        role: "system",
        content: STAKEHOLDER_AGENT_SYSTEM_PROMPT
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

  if (model.startsWith("gpt-5")) {
    requestBody.reasoning = { effort: "minimal" };
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

function sanitizeStakeholderAgentResult(
  result: Partial<StakeholderAgentResult>,
  context: ProductContext,
  model: string,
  orgChart: OrgChart
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
    agentMode: "gpt",
    model,
    primaryBuyers,
    influencers,
    executiveApprovers,
    pathToDecisionMaker,
    summary: typeof result.summary === "string" ? result.summary : "",
    reasoning: {
      scoringPriorities: sanitizeStringArray(result.reasoning?.scoringPriorities),
      primaryBuyerCriteria: sanitizeString(result.reasoning?.primaryBuyerCriteria),
      influencerCriteria: sanitizeString(result.reasoning?.influencerCriteria),
      executiveApproverCriteria: sanitizeString(result.reasoning?.executiveApproverCriteria)
    }
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
