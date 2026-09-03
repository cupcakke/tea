module main

import crypto.sha256
import db.sqlite
import json
import math
import net.http
import os
import rand
import strconv
import strings
import time
import vweb

struct Config {
	base_url               string
	api_key                string
	model                  string
	embedding_model        string
	db_path                string
	workspace_root         string
	knowledge_root         string
	port                   int
	max_model_tokens       int
	default_token_budget   int
	max_steps              int
	step_retry_limit       int
	tenant_header          string
	public_origin          string
	allow_write_tools      bool
	enable_learning        bool
	enable_meta_agent      bool
	meta_interval_seconds  int
	output_classifier_url  string
	diagnostic_timeout_ms  int
	max_state_bytes        int
	max_observation_bytes  int
	max_tool_output_bytes  int
	max_diagnostic_tasks   int
	deliberation_lease_ms  i64
}

struct App {
	vweb.Context
mut:
	cfg Config
}

struct ErrorResponse {
	error string
}

struct ApiChatRequest {
	message          string
	task             string
	session_id       string
	wait_ms          int
	max_steps        int
	token_budget     int
	verifier_json    string
	success_criteria []string
	constraints      []string
	messages         []ChatMessage
	autonomous       bool
}

struct CreateSessionResponse {
	session_id string
	status     string
	answer     string
	state_url  string
	events_url string
}

struct ChatMessage {
	role    string
	content string
}

struct OpenAIChatRequest {
	model       string
	messages    []ChatMessage
	stream      bool
	temperature f64
	max_tokens  int
}

struct OpenAICompatibleResponse {
	id      string
	object  string
	created int
	model   string
	choices []OpenAIChoice
	usage   Usage
}

struct OpenAIChoice {
	index         int
	message       ChatMessage
	finish_reason string
}

struct StreamOptions {
	include_usage bool
}

struct ChatCompletionRequest {
	model             string
	messages          []ChatMessage
	stream            bool
	stream_options    StreamOptions
	temperature       f64
	top_p             f64
	max_tokens        int
	frequency_penalty f64
	presence_penalty  f64
	seed              int
}

struct ChatCompletionScoringRequest {
	model        string
	messages     []ChatMessage
	stream       bool
	temperature  f64
	top_p        f64
	max_tokens   int
	logprobs     bool
	top_logprobs int
	seed         int
}

struct Usage {
	prompt_tokens     int
	completion_tokens int
	total_tokens      int
}

struct ChatCompletionChunk {
	choices []ChatChunkChoice
	usage   Usage
}

struct ChatChunkChoice {
	delta         ChatDelta
	finish_reason string
}

struct ChatDelta {
	content string
}

struct ChatCompletionResponse {
	choices []ChatChoice
	usage   Usage
}

struct ChatChoice {
	message       ChatMessage
	finish_reason string
}

struct ModelResult {
	content           string
	prompt_tokens     int
	completion_tokens int
	total_tokens      int
}

struct EmbeddingRequest {
	model string
	input string
}

struct EmbeddingData {
	embedding []f64
	index     int
}

struct EmbeddingResponse {
	data  []EmbeddingData
	usage Usage
}

struct SessionRecord {
	id               string
	tenant_id        string
	spec_json        string
	state_json       string
	wm_json          string
	status           string
	created_at_ms    i64
	updated_at_ms    i64
	lease_expires_ms i64
	last_step        int
	max_steps        int
	token_budget     int
	tokens_used      int
	observation_json string
	workspace_path   string
	verifier_json    string
	error_message    string
}

struct ActionRecord {
	id          string
	tenant_id   string
	session_id  string
	step        int
	action_json string
	status      string
}

struct Skill {
	id             string
	tenant_id      string
	name           string
	description    string
	trigger_json   string
	body           string
	embedding_json string
	version        int
	enabled        int
	tests_json     string
	updated_at_ms  i64
}

struct ActionCommand {
	tool                   string
	arguments              JsonNode
	idempotency_key        string
	requires_authorization bool
}

struct StepDecision {
	state_patch       JsonNode
	wm_patch          JsonNode
	action            ActionCommand
	terminal          bool
	selected_skill_id string
	cognition_tokens  [][]f64
	transition_gate   f64
	subgoals          []string
	schema_json       string
}

struct ValidatedTransition {
	new_state_json string
	new_wm_json    string
	action_json    string
	cognition_json string
}

struct ToolExecution {
	ok           bool
	observation  JsonNode
	receipt      JsonNode
	terminal     bool
	final_answer string
}

struct VerifierResult {
	ok      bool
	details string
}

struct ScoredId {
	id    string
	score f64
}

struct DenseScore {
	id    string
	score f64
}

struct ScoredToken {
	token   string
	logprob f64
	top     map[string]f64
}

struct ScoredCompletion {
	content string
	tokens  []ScoredToken
}

struct ClassifierRequest {
	tenant_id string
	text      string
}

struct ClassifierResponse {
	allowed bool
	reason  string
}

struct ToolAuthRequest {
	tool_name          string
	path_prefix        string
	expires_in_seconds int
}

struct SkillPatchValidation {
	ok      bool
	details string
}

enum JsonKind {
	null_value
	bool_value
	number_value
	string_value
	array_value
	object_value
}

struct JsonNode {
	kind JsonKind
	s    string
	n    f64
	b    bool
	arr  []JsonNode
	obj  map[string]JsonNode
}

struct JsonParser {
	text string
mut:
	i int
}

fn env_int(name string, default_value int) int {
	raw := os.getenv(name).trim_space()
	if raw == '' {
		return default_value
	}
	return strconv.atoi(raw) or { default_value }
}

fn env_bool(name string, default_value bool) bool {
	raw := os.getenv(name).trim_space().to_lower()
	if raw == '' {
		return default_value
	}
	match raw {
		'1', 'true', 'yes', 'on' {
			return true
		}
		'0', 'false', 'no', 'off' {
			return false
		}
		else {
			return default_value
		}
	}
}

fn strip_trailing_slash(value string) string {
	mut out := value
	for out.ends_with('/') && out.len > 1 {
		out = out[..out.len - 1]
	}
	return out
}

fn load_config() Config {
	api_key := os.getenv('MODULAR_API_KEY').trim_space()
	if api_key == '' {
		panic('MODULAR_API_KEY is required')
	}
	workspace_root := os.getenv('WORKSPACE_ROOT').trim_space()
	knowledge_root := os.getenv('KNOWLEDGE_ROOT').trim_space()
	mut effective_workspace := workspace_root
	mut effective_knowledge := knowledge_root
	if effective_workspace == '' {
		effective_workspace = os.join_path(os.getwd(), 'workspace')
	}
	if effective_knowledge == '' {
		effective_knowledge = os.join_path(os.getwd(), 'knowledge')
	}
	os.mkdir_all(effective_workspace) or { panic(err) }
	os.mkdir_all(effective_knowledge) or { panic(err) }
	raw_base := os.getenv('MODULAR_BASE_URL').trim_space()
	effective_base := if raw_base == '' { 'https://api.modular.com/v1' } else { strip_trailing_slash(raw_base) }
	raw_model := os.getenv('MODULAR_MODEL').trim_space()
	effective_model := if raw_model == '' { 'zai-org/glm-5.3' } else { raw_model }
	raw_db := os.getenv('DATABASE_PATH').trim_space()
	effective_db := if raw_db == '' { os.join_path(os.getwd(), 'agent_runtime.sqlite') } else { raw_db }
	raw_tenant := os.getenv('TENANT_HEADER').trim_space()
	effective_tenant := if raw_tenant == '' { 'X-Tenant-ID' } else { raw_tenant }
	raw_origin := os.getenv('PUBLIC_ORIGIN').trim_space()
	effective_origin := if raw_origin == '' { '*' } else { raw_origin }
	return Config{
		base_url: effective_base
		api_key: api_key
		model: effective_model
		embedding_model: os.getenv('EMBEDDING_MODEL').trim_space()
		db_path: effective_db
		workspace_root: effective_workspace
		knowledge_root: effective_knowledge
		port: env_int('PORT', 8080)
		max_model_tokens: env_int('MAX_MODEL_TOKENS', 100000)
		default_token_budget: env_int('DEFAULT_TOKEN_BUDGET', 4000000)
		max_steps: env_int('MAX_AGENT_STEPS', 1000)
		step_retry_limit: env_int('STEP_RETRY_LIMIT', 3)
		tenant_header: effective_tenant
		public_origin: effective_origin
		allow_write_tools: env_bool('ALLOW_WRITE_TOOLS', false)
		enable_learning: env_bool('ENABLE_LEARNING', true)
		enable_meta_agent: env_bool('ENABLE_META_AGENT', true)
		meta_interval_seconds: env_int('META_INTERVAL_SECONDS', 60)
		output_classifier_url: os.getenv('OUTPUT_CLASSIFIER_URL').trim_space()
		diagnostic_timeout_ms: env_int('DIAGNOSTIC_TIMEOUT_MS', 120000)
		max_state_bytes: env_int('MAX_STATE_BYTES', 65536)
		max_observation_bytes: env_int('MAX_OBSERVATION_BYTES', 32768)
		max_tool_output_bytes: env_int('MAX_TOOL_OUTPUT_BYTES', 65536)
		max_diagnostic_tasks: env_int('MAX_DIAGNOSTIC_TASKS', 6)
		deliberation_lease_ms: 45000
	}
}

fn now_ms() i64 {
	return time.now().unix_milli()
}

fn now_seconds() i64 {
	return time.now().unix()
}

fn new_id(prefix string) string {
	return '${prefix}_${now_ms()}_${rand.u64()}_${rand.u64()}'
}

fn sha_hex(value string) string {
	return sha256.hexhash(value)
}

fn truncate_text(value string, max_len int) string {
	if max_len <= 0 {
		return ''
	}
	if value.len <= max_len {
		return value
	}
	return value[..max_len]
}

fn estimate_tokens(value string) int {
	if value.len == 0 {
		return 0
	}
	return (value.len / 4) + 1
}

fn is_alpha(c u8) bool {
	return (c >= `a` && c <= `z`) || (c >= `A` && c <= `Z`)
}

fn is_digit(c u8) bool {
	return c >= `0` && c <= `9`
}

fn is_alnum(c u8) bool {
	return is_alpha(c) || is_digit(c)
}

fn sanitize_identifier(value string) !string {
	trimmed := value.trim_space()
	if trimmed.len == 0 || trimmed.len > 96 {
		return error('invalid identifier length')
	}
	for i := 0; i < trimmed.len; i++ {
		c := trimmed[i]
		if !(is_alnum(c) || c == `_` || c == `-` || c == `.`) {
			return error('invalid identifier character')
		}
	}
	return trimmed
}

fn jnull() JsonNode {
	return JsonNode{
		kind: .null_value
	}
}

fn jbool(value bool) JsonNode {
	return JsonNode{
		kind: .bool_value
		b: value
	}
}

fn jnum(value f64) JsonNode {
	return JsonNode{
		kind: .number_value
		n: value
	}
}

fn jstr(value string) JsonNode {
	return JsonNode{
		kind: .string_value
		s: value
	}
}

fn jarr(value []JsonNode) JsonNode {
	return JsonNode{
		kind: .array_value
		arr: value
	}
}

fn jobj(value map[string]JsonNode) JsonNode {
	return JsonNode{
		kind: .object_value
		obj: value
	}
}

fn parse_json_node(text string) !JsonNode {
	mut parser := JsonParser{
		text: text
	}
	parser.skip_ws()
	value := parser.parse_value()!
	parser.skip_ws()
	if parser.i != parser.text.len {
		return error('trailing content after json')
	}
	return value
}

fn (mut parser JsonParser) skip_ws() {
	for parser.i < parser.text.len {
		c := parser.text[parser.i]
		if c == ` ` || c == `\n` || c == `\r` || c == `\t` {
			parser.i++
		} else {
			break
		}
	}
}

fn (mut parser JsonParser) consume_literal(literal string) bool {
	if parser.i + literal.len > parser.text.len {
		return false
	}
	if parser.text[parser.i..parser.i + literal.len] == literal {
		parser.i += literal.len
		return true
	}
	return false
}

fn (mut parser JsonParser) parse_value() !JsonNode {
	parser.skip_ws()
	if parser.i >= parser.text.len {
		return error('unexpected end of json')
	}
	c := parser.text[parser.i]
	if c == `n` {
		if parser.consume_literal('null') {
			return jnull()
		}
		return error('invalid null literal')
	}
	if c == `t` {
		if parser.consume_literal('true') {
			return jbool(true)
		}
		return error('invalid true literal')
	}
	if c == `f` {
		if parser.consume_literal('false') {
			return jbool(false)
		}
		return error('invalid false literal')
	}
	if c == `"` {
		return jstr(parser.parse_string_raw()!)
	}
	if c == `[` {
		return parser.parse_array()
	}
	if c == `{` {
		return parser.parse_object()
	}
	if c == `-` || is_digit(c) {
		return parser.parse_number()
	}
	return error('unexpected json token')
}

fn hex_value(c u8) int {
	if c >= `0` && c <= `9` {
		return int(c - `0`)
	}
	if c >= `a` && c <= `f` {
		return int(c - `a`) + 10
	}
	if c >= `A` && c <= `F` {
		return int(c - `A`) + 10
	}
	return -1
}

fn utf8_from_codepoint(cp int) string {
	mut bytes := []u8{}
	if cp <= 0x7f {
		bytes << u8(cp)
	} else if cp <= 0x7ff {
		bytes << u8(0xc0 | ((cp >> 6) & 0x1f))
		bytes << u8(0x80 | (cp & 0x3f))
	} else if cp <= 0xffff {
		bytes << u8(0xe0 | ((cp >> 12) & 0x0f))
		bytes << u8(0x80 | ((cp >> 6) & 0x3f))
		bytes << u8(0x80 | (cp & 0x3f))
	} else {
		bytes << u8(0xf0 | ((cp >> 18) & 0x07))
		bytes << u8(0x80 | ((cp >> 12) & 0x3f))
		bytes << u8(0x80 | ((cp >> 6) & 0x3f))
		bytes << u8(0x80 | (cp & 0x3f))
	}
	return bytes.bytestr()
}

fn (mut parser JsonParser) parse_hex4() !int {
	if parser.i + 4 > parser.text.len {
		return error('incomplete unicode escape')
	}
	mut value := 0
	for _ in 0 .. 4 {
		h := hex_value(parser.text[parser.i])
		if h < 0 {
			return error('invalid unicode escape')
		}
		value = (value << 4) | h
		parser.i++
	}
	return value
}

fn (mut parser JsonParser) parse_string_raw() !string {
	if parser.i >= parser.text.len || parser.text[parser.i] != `"` {
		return error('expected string')
	}
	parser.i++
	mut builder := strings.new_builder(32)
	for parser.i < parser.text.len {
		c := parser.text[parser.i]
		if c == `"` {
			parser.i++
			return builder.str()
		}
		if c == `\\` {
			parser.i++
			if parser.i >= parser.text.len {
				return error('incomplete escape')
			}
			esc := parser.text[parser.i]
			parser.i++
			match esc {
				`"` {
					builder.write_u8(`"`)
				}
				`\\` {
					builder.write_u8(`\\`)
				}
				`/` {
					builder.write_u8(`/`)
				}
				`b` {
					builder.write_u8(8)
				}
				`f` {
					builder.write_u8(12)
				}
				`n` {
					builder.write_u8(`\n`)
				}
				`r` {
					builder.write_u8(`\r`)
				}
				`t` {
					builder.write_u8(`\t`)
				}
				`u` {
					mut cp := parser.parse_hex4()!
					if cp >= 0xd800 && cp <= 0xdbff {
						if parser.i + 6 <= parser.text.len && parser.text[parser.i] == `\\`
							&& parser.text[parser.i + 1] == `u` {
							parser.i += 2
							low := parser.parse_hex4()!
							if low < 0xdc00 || low > 0xdfff {
								return error('invalid surrogate pair')
							}
							cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00)
						} else {
							return error('missing low surrogate')
						}
					}
					builder.write_string(utf8_from_codepoint(cp))
				}
				else {
					return error('invalid escape')
				}
			}
		} else {
			if c < 0x20 {
				return error('control character in string')
			}
			builder.write_u8(c)
			parser.i++
		}
	}
	return error('unterminated string')
}

fn (mut parser JsonParser) parse_number() !JsonNode {
	start := parser.i
	if parser.text[parser.i] == `-` {
		parser.i++
	}
	if parser.i >= parser.text.len {
		return error('invalid number')
	}
	if parser.text[parser.i] == `0` {
		parser.i++
	} else {
		if !is_digit(parser.text[parser.i]) {
			return error('invalid number')
		}
		for parser.i < parser.text.len && is_digit(parser.text[parser.i]) {
			parser.i++
		}
	}
	if parser.i < parser.text.len && parser.text[parser.i] == `.` {
		parser.i++
		if parser.i >= parser.text.len || !is_digit(parser.text[parser.i]) {
			return error('invalid number fraction')
		}
		for parser.i < parser.text.len && is_digit(parser.text[parser.i]) {
			parser.i++
		}
	}
	if parser.i < parser.text.len && (parser.text[parser.i] == `e` || parser.text[parser.i] == `E`) {
		parser.i++
		if parser.i < parser.text.len && (parser.text[parser.i] == `+` || parser.text[parser.i] == `-`) {
			parser.i++
		}
		if parser.i >= parser.text.len || !is_digit(parser.text[parser.i]) {
			return error('invalid number exponent')
		}
		for parser.i < parser.text.len && is_digit(parser.text[parser.i]) {
			parser.i++
		}
	}
	raw := parser.text[start..parser.i]
	number := strconv.atof64(raw) or { return error('invalid number value') }
	return jnum(number)
}

fn (mut parser JsonParser) parse_array() !JsonNode {
	if parser.text[parser.i] != `[` {
		return error('expected array')
	}
	parser.i++
	mut values := []JsonNode{}
	parser.skip_ws()
	if parser.i < parser.text.len && parser.text[parser.i] == `]` {
		parser.i++
		return jarr(values)
	}
	for {
		value := parser.parse_value()!
		values << value
		parser.skip_ws()
		if parser.i >= parser.text.len {
			return error('unterminated array')
		}
		if parser.text[parser.i] == `]` {
			parser.i++
			break
		}
		if parser.text[parser.i] != `,` {
			return error('expected array comma')
		}
		parser.i++
	}
	return jarr(values)
}

fn (mut parser JsonParser) parse_object() !JsonNode {
	if parser.text[parser.i] != `{` {
		return error('expected object')
	}
	parser.i++
	mut values := map[string]JsonNode{}
	parser.skip_ws()
	if parser.i < parser.text.len && parser.text[parser.i] == `}` {
		parser.i++
		return jobj(values)
	}
	for {
		parser.skip_ws()
		key := parser.parse_string_raw()!
		parser.skip_ws()
		if parser.i >= parser.text.len || parser.text[parser.i] != `:` {
			return error('expected object colon')
		}
		parser.i++
		value := parser.parse_value()!
		values[key] = value
		parser.skip_ws()
		if parser.i >= parser.text.len {
			return error('unterminated object')
		}
		if parser.text[parser.i] == `}` {
			parser.i++
			break
		}
		if parser.text[parser.i] != `,` {
			return error('expected object comma')
		}
		parser.i++
	}
	return jobj(values)
}

fn hex_digit(value int) u8 {
	if value >= 0 && value <= 9 {
		return u8(int(`0`) + value)
	}
	return u8(int(`a`) + value - 10)
}

fn json_escape(value string) string {
	mut builder := strings.new_builder(value.len + 16)
	for i := 0; i < value.len; i++ {
		c := value[i]
		match c {
			`"` {
				builder.write_string('\\"')
			}
			`\\` {
				builder.write_string('\\\\')
			}
			`\n` {
				builder.write_string('\\n')
			}
			`\r` {
				builder.write_string('\\r')
			}
			`\t` {
				builder.write_string('\\t')
			}
			else {
				if c < 0x20 {
					builder.write_string('\\u00')
					builder.write_u8(hex_digit(int(c) >> 4))
					builder.write_u8(hex_digit(int(c) & 15))
				} else {
					builder.write_u8(c)
				}
			}
		}
	}
	return builder.str()
}

fn format_json_number(value f64) string {
	if value != value {
		return '0'
	}
	if value > 1.0e308 || value < -1.0e308 {
		return '0'
	}
	raw := '${value}'
	if raw == 'inf' || raw == '+inf' || raw == '-inf' {
		return '0'
	}
	return raw
}

fn serialize_json_node(node JsonNode) string {
	match node.kind {
		.null_value {
			return 'null'
		}
		.bool_value {
			if node.b {
				return 'true'
			}
			return 'false'
		}
		.number_value {
			return format_json_number(node.n)
		}
		.string_value {
			return '"' + json_escape(node.s) + '"'
		}
		.array_value {
			mut builder := strings.new_builder(64)
			builder.write_u8(`[`)
			for index, value in node.arr {
				if index > 0 {
					builder.write_u8(`,`)
				}
				builder.write_string(serialize_json_node(value))
			}
			builder.write_u8(`]`)
			return builder.str()
		}
		.object_value {
			mut keys := node.obj.keys()
			keys.sort()
			mut builder := strings.new_builder(64)
			builder.write_u8(`{`)
			for index, key in keys {
				if index > 0 {
					builder.write_u8(`,`)
				}
				builder.write_u8(`"`)
				builder.write_string(json_escape(key))
				builder.write_string('":')
				builder.write_string(serialize_json_node(node.obj[key]))
			}
			builder.write_u8(`}`)
			return builder.str()
		}
	}
}

fn node_get(node JsonNode, key string) ?JsonNode {
	if node.kind != .object_value {
		return none
	}
	if !(key in node.obj) {
		return none
	}
	return node.obj[key]
}

fn node_as_string(node JsonNode) !string {
	match node.kind {
		.string_value {
			return node.s
		}
		.number_value {
			return format_json_number(node.n)
		}
		.bool_value {
			if node.b {
				return 'true'
			}
			return 'false'
		}
		else {
			return error('expected scalar string')
		}
	}
}

fn node_as_f64(node JsonNode) !f64 {
	match node.kind {
		.number_value {
			return node.n
		}
		.string_value {
			return strconv.atof64(node.s) or { return error('invalid numeric string') }
		}
		else {
			return error('expected number')
		}
	}
}

fn node_as_int(node JsonNode) !int {
	match node.kind {
		.number_value {
			return int(node.n)
		}
		.string_value {
			return strconv.atoi(node.s) or { return error('invalid integer string') }
		}
		else {
			return error('expected integer')
		}
	}
}

fn node_as_bool(node JsonNode) !bool {
	match node.kind {
		.bool_value {
			return node.b
		}
		.string_value {
			lower := node.s.to_lower()
			if lower == 'true' || lower == '1' || lower == 'yes' {
				return true
			}
			if lower == 'false' || lower == '0' || lower == 'no' {
				return false
			}
			return error('invalid boolean string')
		}
		else {
			return error('expected boolean')
		}
	}
}

fn optional_string_field(node JsonNode, key string, default_value string) string {
	value := node_get(node, key) or { return default_value }
	return node_as_string(value) or { default_value }
}

fn required_string_field(node JsonNode, key string) !string {
	value := node_get(node, key) or { return error('missing required field ' + key) }
	return node_as_string(value)
}

fn optional_bool_field(node JsonNode, key string, default_value bool) bool {
	value := node_get(node, key) or { return default_value }
	return node_as_bool(value) or { default_value }
}

fn optional_int_field(node JsonNode, key string, default_value int) int {
	value := node_get(node, key) or { return default_value }
	return node_as_int(value) or { default_value }
}

fn optional_f64_field(node JsonNode, key string, default_value f64) f64 {
	value := node_get(node, key) or { return default_value }
	return node_as_f64(value) or { default_value }
}

fn node_string_array(node JsonNode) ![]string {
	if node.kind != .array_value {
		return error('expected string array')
	}
	mut values := []string{}
	for item in node.arr {
		values << node_as_string(item)!
	}
	return values
}

fn string_array_node(values []string) JsonNode {
	mut arr := []JsonNode{}
	for value in values {
		arr << jstr(value)
	}
	return jarr(arr)
}

fn extract_first_json_object(text string) !string {
	mut start := -1
	mut depth := 0
	mut in_string := false
	mut escaping := false
	for i := 0; i < text.len; i++ {
		c := text[i]
		if start < 0 {
			if c == `{` {
				start = i
				depth = 1
				in_string = false
				escaping = false
			}
			continue
		}
		if in_string {
			if escaping {
				escaping = false
			} else if c == `\\` {
				escaping = true
			} else if c == `"` {
				in_string = false
			}
		} else {
			if c == `"` {
				in_string = true
			} else if c == `{` {
				depth++
			} else if c == `}` {
				depth--
				if depth == 0 {
					return text[start..i + 1]
				}
			}
		}
	}
	return error('no complete json object found')
}

fn parse_int_default(value string, default_value int) int {
	return strconv.atoi(value) or { default_value }
}

fn parse_i64_default(value string, default_value i64) i64 {
	return strconv.parse_int(value, 10, 64) or { default_value }
}

fn row_get(row sqlite.Row, index int) string {
	if index < 0 || index >= row.vals.len {
		return ''
	}
	return row.vals[index]
}

fn open_db(cfg Config) !sqlite.DB {
	mut db := sqlite.connect(cfg.db_path)!
	db.exec('pragma journal_mode=WAL')!
	db.exec('pragma foreign_keys=ON')!
	db.exec('pragma busy_timeout=5000')!
	return db
}

fn init_db(cfg Config) ! {
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	db.exec('create table if not exists tenants (id text primary key, created_at_ms integer not null)')!
	db.exec('create table if not exists sessions (id text not null, tenant_id text not null, spec_json text not null, state_json text not null, wm_json text not null, status text not null, created_at_ms integer not null, updated_at_ms integer not null, lease_expires_ms integer not null, last_step integer not null, max_steps integer not null, token_budget integer not null, tokens_used integer not null, observation_json text not null, workspace_path text not null, verifier_json text not null, error text not null, primary key(id, tenant_id))')!
	db.exec('create index if not exists idx_sessions_status on sessions(status, lease_expires_ms)')!
	db.exec('create table if not exists checkpoints (id text primary key, tenant_id text not null, session_id text not null, step integer not null, phase text not null, state_json text not null, wm_json text not null, observation_json text not null, patch_json text not null, action_json text not null, created_at_ms integer not null, hash text not null)')!
	db.exec('create index if not exists idx_checkpoints_session on checkpoints(tenant_id, session_id, step, phase)')!
	db.exec('create table if not exists observations (id text primary key, tenant_id text not null, session_id text not null, step integer not null, obs_json text not null, created_at_ms integer not null)')!
	db.exec('create table if not exists actions (id text primary key, tenant_id text not null, session_id text not null, step integer not null, action_json text not null, idempotency_key text not null, status text not null, result_json text not null, error text not null, created_at_ms integer not null, updated_at_ms integer not null)')!
	db.exec('create index if not exists idx_actions_status on actions(status, created_at_ms)')!
	db.exec('create index if not exists idx_actions_idempotency on actions(tenant_id, session_id, idempotency_key)')!
	db.exec('create table if not exists raw_traces (id text primary key, tenant_id text not null, session_id text not null, trace_json text not null, created_at_ms integer not null, hash text not null)')!
	db.exec('create index if not exists idx_raw_traces_session on raw_traces(tenant_id, session_id, created_at_ms)')!
	db.exec('create table if not exists skills (id text not null, tenant_id text not null, name text not null, description text not null, trigger_json text not null, body text not null, embedding_json text not null, version integer not null, enabled integer not null, tests_json text not null, created_at_ms integer not null, updated_at_ms integer not null, primary key(id, tenant_id))')!
	db.exec('create index if not exists idx_skills_tenant_enabled on skills(tenant_id, enabled, updated_at_ms)')!
	db.exec('create virtual table if not exists skills_fts using fts5(id unindexed, tenant_id unindexed, name, description, body)')!
	db.exec('create table if not exists skill_events (id text primary key, tenant_id text not null, skill_id text not null, event_json text not null, created_at_ms integer not null)')!
	db.exec('create table if not exists cognition_frames (id text primary key, tenant_id text not null, session_id text not null, step integer not null, cognition_json text not null, generated_at_ms integer not null)')!
	db.exec('create index if not exists idx_cognition_session on cognition_frames(tenant_id, session_id, generated_at_ms)')!
	db.exec('create table if not exists reflections (id text primary key, tenant_id text not null, session_id text not null, reflection_json text not null, verifier_ok integer not null, processed integer not null, created_at_ms integer not null)')!
	db.exec('create index if not exists idx_reflections_processed on reflections(processed, created_at_ms)')!
	db.exec('create table if not exists distillation_tokens (id text primary key, tenant_id text not null, session_id text not null, step integer not null, position integer not null, student_token text not null, teacher_token text not null, student_logprob real not null, teacher_logprob real not null, reverse_kl real not null, teacher_top_json text not null, student_top_json text not null, created_at_ms integer not null)')!
	db.exec('create table if not exists policy_weights (tenant_id text not null, key text not null, weight real not null, updated_at_ms integer not null, primary key(tenant_id, key))')!
	db.exec('create table if not exists knowledge_versions (id text primary key, tenant_id text not null, path text not null, diff text not null, created_at_ms integer not null)')!
	db.exec('create table if not exists tool_authorizations (id text primary key, tenant_id text not null, session_id text not null, tool_name text not null, path_prefix text not null, expires_at_ms integer not null, created_at_ms integer not null)')!
	db.exec('create index if not exists idx_tool_authorizations on tool_authorizations(tenant_id, session_id, tool_name, expires_at_ms)')!
	db.exec('create table if not exists diagnostic_tasks (id text not null, tenant_id text not null, task text not null, verifier_json text not null, created_at_ms integer not null, primary key(id, tenant_id))')!
	db.exec_param_many('insert or ignore into tenants(id, created_at_ms) values(?, ?)', ['global', now_ms().str()])!
	db.exec_param_many('insert or ignore into tenants(id, created_at_ms) values(?, ?)', ['default', now_ms().str()])!
	ensure_builtin_skills(mut db)!
	ensure_builtin_diagnostics(mut db)!
}

fn ensure_tenant(mut db sqlite.DB, tenant_id string) ! {
	safe := sanitize_identifier(tenant_id)!
	db.exec_param_many('insert or ignore into tenants(id, created_at_ms) values(?, ?)', [safe, now_ms().str()])!
}

fn ensure_builtin_skills(mut db sqlite.DB) ! {
	seed_skill(mut db, 'global', 'builtin_final_answer', 'final answer delivery', 'Deliver a verified final user-visible answer and terminate only when verifiers can pass.', jobj({
		'tools': string_array_node(['final_answer'])
	}), 'Use final_answer only when the structured state and latest observation contain enough verified evidence to satisfy the task. The arguments object must contain a concise answer string. Never include hidden reasoning or transcripts.', jarr([
		jobj({
			'task': jstr('Answer with the word ready.')
			'verifier': jobj({
				'type': jstr('answer_contains')
				'value': jstr('ready')
			})
		})
	]))!
	seed_skill(mut db, 'global', 'builtin_file_append_unique', 'append file unique lines', 'Append lines to a workspace file with exact-line deduplication and durable receipts.', jobj({
		'tools': string_array_node(['append_file'])
	}), 'Use append_file for line-oriented durable notes or artifacts. Set unique to true when each exact line must appear at most once. The path is relative to the authorized workspace. The arguments object accepts path, content or lines, and unique.', jarr([]JsonNode{}))!
	seed_skill(mut db, 'global', 'builtin_file_replace_lines', 'replace file line range', 'Replace an inclusive one-based line range inside a workspace file atomically.', jobj({
		'tools': string_array_node(['replace_lines'])
	}), 'Use replace_lines when a deterministic line interval must be replaced. Arguments require path, start_line, end_line, and replacement or lines. Lines are one-based and inclusive.', jarr([]JsonNode{}))!
	seed_skill(mut db, 'global', 'builtin_file_check_lines', 'check exact file lines', 'Check membership for a batch of exact lines in a workspace file.', jobj({
		'tools': string_array_node(['check_lines'])
	}), 'Use check_lines to verify whether exact lines already exist in a workspace file. Arguments require path and lines. The result maps each requested exact line to a boolean.', jarr([]JsonNode{}))!
}

fn ensure_builtin_diagnostics(mut db sqlite.DB) ! {
	seed_diagnostic_task(mut db, 'global', 'diag_final_answer_ready', 'Return the exact word ready as the final answer.', jobj({
		'type': jstr('answer_contains')
		'value': jstr('ready')
	}))!
	seed_diagnostic_task(mut db, 'global', 'diag_file_append_test', 'Append a marker line to notes.txt and deliver final answer.', jobj({
		'type': jstr('all')
		'checks': jarr([
			jobj({
				'type': jstr('file_contains_lines')
				'path': jstr('notes.txt')
				'lines': string_array_node(['diagnostic_pass'])
			}),
			jobj({
				'type': jstr('final_answer_nonempty')
			})
		])
	}))!
}

fn seed_skill(mut db sqlite.DB, tenant_id string, id string, name string, description string, trigger JsonNode, body string, tests JsonNode) ! {
	rows := db.exec_param_many('select id from skills where tenant_id = ? and id = ?', [tenant_id, id])!
	if rows.len > 0 {
		refresh_skill_fts(mut db, tenant_id, id)!
		return
	}
	created := now_ms().str()
	db.exec_param_many('insert into skills(id, tenant_id, name, description, trigger_json, body, embedding_json, version, enabled, tests_json, created_at_ms, updated_at_ms) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
		id,
		tenant_id,
		name,
		description,
		serialize_json_node(trigger),
		body,
		'[]',
		'1',
		'1',
		serialize_json_node(tests),
		created,
		created,
	])!
	refresh_skill_fts(mut db, tenant_id, id)!
}

fn seed_diagnostic_task(mut db sqlite.DB, tenant_id string, id string, task string, verifier JsonNode) ! {
	db.exec_param_many('insert or ignore into diagnostic_tasks(id, tenant_id, task, verifier_json, created_at_ms) values(?, ?, ?, ?, ?)', [
		id,
		tenant_id,
		task,
		serialize_json_node(verifier),
		now_ms().str(),
	])!
}

fn refresh_skill_fts(mut db sqlite.DB, tenant_id string, id string) ! {
	db.exec_param_many('delete from skills_fts where tenant_id = ? and id = ?', [tenant_id, id])!
	rows := db.exec_param_many('select id, tenant_id, name, description, body from skills where tenant_id = ? and id = ? and enabled = ?', [
		tenant_id,
		id,
		'1',
	])!
	if rows.len == 0 {
		return
	}
	row := rows[0]
	db.exec_param_many('insert into skills_fts(id, tenant_id, name, description, body) values(?, ?, ?, ?, ?)', [
		row_get(row, 0),
		row_get(row, 1),
		row_get(row, 2),
		row_get(row, 3),
		row_get(row, 4),
	])!
}

fn session_from_row(row sqlite.Row) SessionRecord {
	return SessionRecord{
		id: row_get(row, 0)
		tenant_id: row_get(row, 1)
		spec_json: row_get(row, 2)
		state_json: row_get(row, 3)
		wm_json: row_get(row, 4)
		status: row_get(row, 5)
		created_at_ms: parse_i64_default(row_get(row, 6), 0)
		updated_at_ms: parse_i64_default(row_get(row, 7), 0)
		lease_expires_ms: parse_i64_default(row_get(row, 8), 0)
		last_step: parse_int_default(row_get(row, 9), 0)
		max_steps: parse_int_default(row_get(row, 10), 0)
		token_budget: parse_int_default(row_get(row, 11), 0)
		tokens_used: parse_int_default(row_get(row, 12), 0)
		observation_json: row_get(row, 13)
		workspace_path: row_get(row, 14)
		verifier_json: row_get(row, 15)
		error_message: row_get(row, 16)
	}
}

fn load_session(mut db sqlite.DB, tenant_id string, session_id string) !SessionRecord {
	rows := db.exec_param_many('select id, tenant_id, spec_json, state_json, wm_json, status, created_at_ms, updated_at_ms, lease_expires_ms, last_step, max_steps, token_budget, tokens_used, observation_json, workspace_path, verifier_json, error from sessions where tenant_id = ? and id = ?', [
		tenant_id,
		session_id,
	])!
	if rows.len == 0 {
		return error('session not found')
	}
	return session_from_row(rows[0])
}

fn action_from_row(row sqlite.Row) ActionRecord {
	return ActionRecord{
		id: row_get(row, 0)
		tenant_id: row_get(row, 1)
		session_id: row_get(row, 2)
		step: parse_int_default(row_get(row, 3), 0)
		action_json: row_get(row, 4)
		status: row_get(row, 5)
	}
}

fn load_action(mut db sqlite.DB, action_id string) !ActionRecord {
	rows := db.exec_param_many('select id, tenant_id, session_id, step, action_json, status from actions where id = ?', [action_id])!
	if rows.len == 0 {
		return error('action not found')
	}
	return action_from_row(rows[0])
}

fn skill_from_row(row sqlite.Row) Skill {
	return Skill{
		id: row_get(row, 0)
		tenant_id: row_get(row, 1)
		name: row_get(row, 2)
		description: row_get(row, 3)
		trigger_json: row_get(row, 4)
		body: row_get(row, 5)
		embedding_json: row_get(row, 6)
		version: parse_int_default(row_get(row, 7), 1)
		enabled: parse_int_default(row_get(row, 8), 1)
		tests_json: row_get(row, 9)
		updated_at_ms: parse_i64_default(row_get(row, 10), 0)
	}
}

fn pending_action_count(mut db sqlite.DB, tenant_id string, session_id string) !int {
	rows := db.exec_param_many('select count(*) from actions where tenant_id = ? and session_id = ? and (status = ? or status = ?)', [
		tenant_id,
		session_id,
		'pending',
		'running',
	])!
	if rows.len == 0 {
		return 0
	}
	return parse_int_default(row_get(rows[0], 0), 0)
}

fn insert_checkpoint(mut db sqlite.DB, tenant_id string, session_id string, step int, phase string, state_json string, wm_json string, observation_json string, patch_json string, action_json string) ! {
	hash := sha_hex(tenant_id + session_id + step.str() + phase + state_json + wm_json + observation_json + patch_json + action_json)
	db.exec_param_many('insert into checkpoints(id, tenant_id, session_id, step, phase, state_json, wm_json, observation_json, patch_json, action_json, created_at_ms, hash) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
		new_id('checkpoint'),
		tenant_id,
		session_id,
		step.str(),
		phase,
		state_json,
		wm_json,
		observation_json,
		patch_json,
		action_json,
		now_ms().str(),
		hash,
	])!
}

fn insert_observation(mut db sqlite.DB, tenant_id string, session_id string, step int, observation_json string) ! {
	db.exec_param_many('insert into observations(id, tenant_id, session_id, step, obs_json, created_at_ms) values(?, ?, ?, ?, ?, ?)', [
		new_id('observation'),
		tenant_id,
		session_id,
		step.str(),
		observation_json,
		now_ms().str(),
	])!
}

fn insert_raw_trace(mut db sqlite.DB, tenant_id string, session_id string, trace JsonNode) ! {
	trace_json := serialize_json_node(trace)
	db.exec_param_many('insert into raw_traces(id, tenant_id, session_id, trace_json, created_at_ms, hash) values(?, ?, ?, ?, ?, ?)', [
		new_id('trace'),
		tenant_id,
		session_id,
		trace_json,
		now_ms().str(),
		sha_hex(trace_json),
	])!
}

fn mark_session_failed(cfg Config, tenant_id string, session_id string, message string) ! {
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	db.exec_param_many('update sessions set status = ?, error = ?, lease_expires_ms = 0, observation_json = ?, updated_at_ms = ? where tenant_id = ? and id = ?', [
		'failed',
		truncate_text(message, 4000),
		serialize_json_node(jobj({
			'type': jstr('runtime_failure')
			'message': jstr(truncate_text(message, cfg.max_observation_bytes / 2))
			'at_ms': jnum(f64(now_ms()))
		})),
		now_ms().str(),
		tenant_id,
		session_id,
	])!
}

fn initial_spec_json(task string, req ApiChatRequest, verifier_json string) string {
	return serialize_json_node(jobj({
		'task': jstr(task)
		'created_at_ms': jnum(f64(now_ms()))
		'immutability': jstr('procedural specification is immutable for this session')
		'success_criteria': string_array_node(req.success_criteria)
		'constraints': string_array_node(req.constraints)
		'verifier': parse_json_node(verifier_json) or { jobj(map[string]JsonNode{}) }
		'runtime_contract': jobj({
			'prompt_footprint': jstr('constant size per step')
			'history_policy': jstr('no transcripts or prior reasoning in prompts')
			'state_transition': jstr('validated dictionary merge with null deletion')
		})
	}))
}

fn initial_state_json(task string) string {
	return serialize_json_node(jobj({
		'status': jstr('initialized')
		'task_digest': jstr(sha_hex(task))
		'progress': jobj({
			'completed': jarr([]JsonNode{})
			'remaining': jarr([jstr(task)])
		})
		'environment': jobj(map[string]JsonNode{})
		'answer': jnull()
	}))
}

fn initial_wm_json(task string, constraints []string) string {
	return serialize_json_node(jobj({
		'verified_progress': jarr([]JsonNode{})
		'open_goals': jarr([jstr(task)])
		'unresolved_dependencies': jarr([]JsonNode{})
		'constraints': string_array_node(constraints)
		'environmental_constraints': jarr([]JsonNode{})
		'last_skill_ids': jarr([]JsonNode{})
	}))
}

fn initial_observation_json(task string) string {
	return serialize_json_node(jobj({
		'type': jstr('user_task')
		'content': jstr(task)
		'at_ms': jnum(f64(now_ms()))
	}))
}

fn verifier_from_body(body string, req ApiChatRequest) !string {
	if req.verifier_json.trim_space() != '' {
		node := parse_json_node(req.verifier_json.trim_space())!
		return serialize_json_node(node)
	}
	if body.trim_space() != '' {
		root := parse_json_node(body) or { return '{}' }
		verifier_node := node_get(root, 'verifier') or { return '{}' }
		return serialize_json_node(verifier_node)
	}
	return '{}'
}

fn create_session(cfg Config, tenant_id string, req ApiChatRequest, task string, verifier_json string) !string {
	safe_tenant := sanitize_identifier(tenant_id)!
	session_id := new_id('session')
	workspace := os.join_path(cfg.workspace_root, safe_tenant, session_id)
	os.mkdir_all(workspace)!
	max_steps := if req.max_steps > 0 { req.max_steps } else { cfg.max_steps }
	token_budget := if req.token_budget > 0 { req.token_budget } else { cfg.default_token_budget }
	spec_json := initial_spec_json(task, req, verifier_json)
	state_json := initial_state_json(task)
	wm_json := initial_wm_json(task, req.constraints)
	observation_json := initial_observation_json(task)
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	db.exec('begin immediate transaction')!
	ensure_tenant(mut db, safe_tenant)!
	db.exec_param_many('insert into sessions(id, tenant_id, spec_json, state_json, wm_json, status, created_at_ms, updated_at_ms, lease_expires_ms, last_step, max_steps, token_budget, tokens_used, observation_json, workspace_path, verifier_json, error) values(?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 0, ?, ?, ?, ?)', [
		session_id,
		safe_tenant,
		spec_json,
		state_json,
		wm_json,
		'running',
		now_ms().str(),
		now_ms().str(),
		'0',
		max_steps.str(),
		token_budget.str(),
		observation_json,
		workspace,
		verifier_json,
		'',
	])!
	insert_checkpoint(mut db, safe_tenant, session_id, 0, 'initial', state_json, wm_json, observation_json, '{}', '{}')!
	insert_observation(mut db, safe_tenant, session_id, 0, observation_json)!
	db.exec('commit')!
	return session_id
}

fn continue_session(cfg Config, tenant_id string, session_id string, task string) ! {
	observation_json := serialize_json_node(jobj({
		'type': jstr('user_message')
		'content': jstr(task)
		'at_ms': jnum(f64(now_ms()))
	}))
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	session := load_session(mut db, tenant_id, session_id)!
	if session.status == 'completed' || session.status == 'failed' {
		db.exec_param_many('update sessions set status = ?, error = ?, lease_expires_ms = 0, updated_at_ms = ? where tenant_id = ? and id = ?', [
			'running',
			'',
			now_ms().str(),
			tenant_id,
			session_id,
		])!
	}
	db.exec_param_many('update sessions set observation_json = ?, updated_at_ms = ? where tenant_id = ? and id = ?', [
		observation_json,
		now_ms().str(),
		tenant_id,
		session_id,
	])!
	insert_observation(mut db, tenant_id, session_id, session.last_step, observation_json)!
}

fn latest_final_answer(mut db sqlite.DB, tenant_id string, session_id string) string {
	rows := db.exec_param_many('select result_json from actions where tenant_id = ? and session_id = ? and status = ? order by updated_at_ms desc limit 20', [
		tenant_id,
		session_id,
		'done',
	]) or { return '' }
	for row in rows {
		node := parse_json_node(row_get(row, 0)) or { continue }
		answer := optional_string_field(node, 'final_answer', '')
		if answer.trim_space() != '' {
			return answer
		}
		obs := node_get(node, 'observation') or { continue }
		answer2 := optional_string_field(obs, 'answer', '')
		if answer2.trim_space() != '' {
			return answer2
		}
	}
	return ''
}

fn response_for_session(cfg Config, tenant_id string, session_id string) !CreateSessionResponse {
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	session := load_session(mut db, tenant_id, session_id)!
	answer := latest_final_answer(mut db, tenant_id, session_id)
	return CreateSessionResponse{
		session_id: session_id
		status: session.status
		answer: answer
		state_url: '/api/sessions/${session_id}/state'
		events_url: '/api/sessions/${session_id}/events'
	}
}

fn wait_for_session(cfg Config, tenant_id string, session_id string, wait_ms int) !CreateSessionResponse {
	if wait_ms <= 0 {
		return response_for_session(cfg, tenant_id, session_id)
	}
	deadline := now_ms() + i64(wait_ms)
	for now_ms() < deadline {
		resp := response_for_session(cfg, tenant_id, session_id)!
		if resp.status == 'completed' || resp.status == 'failed' || resp.answer.trim_space() != '' {
			return resp
		}
		time.sleep(100 * time.millisecond)
	}
	return response_for_session(cfg, tenant_id, session_id)
}

fn extract_task_from_request(req ApiChatRequest) string {
	if req.task.trim_space() != '' {
		return req.task.trim_space()
	}
	if req.message.trim_space() != '' {
		return req.message.trim_space()
	}
	for idx := req.messages.len; idx > 0; idx-- {
		msg := req.messages[idx - 1]
		if msg.role == 'user' && msg.content.trim_space() != '' {
			return msg.content.trim_space()
		}
	}
	if req.messages.len > 0 {
		return req.messages[req.messages.len - 1].content.trim_space()
	}
	return ''
}

fn call_model_stream(cfg Config, messages []ChatMessage, temperature f64, max_tokens int, freq_penalty f64, pres_penalty f64) !ModelResult {
	payload := ChatCompletionRequest{
		model: cfg.model
		messages: messages
		stream: true
		stream_options: StreamOptions{
			include_usage: true
		}
		temperature: temperature
		top_p: 1.0
		max_tokens: max_tokens
		frequency_penalty: freq_penalty
		presence_penalty: pres_penalty
		seed: 1234
	}
	body := json.encode(payload)
	mut req := http.Request{
		method: .post
		url: cfg.base_url + '/chat/completions'
		data: body
	}
	req.add_header(.content_type, 'application/json')
	req.add_header(.authorization, 'Bearer ' + cfg.api_key)
	resp := req.do()!
	if resp.status_code < 200 || resp.status_code >= 300 {
		return error('model api error ${resp.status_code}: ' + truncate_text(resp.body, 2000))
	}
	return parse_model_response(resp.body)!
}

fn parse_model_response(body string) !ModelResult {
	if body.contains('data:') {
		mut builder := strings.new_builder(body.len)
		mut usage := Usage{}
		for line in body.split_into_lines() {
			trimmed := line.trim_space()
			if !trimmed.starts_with('data:') {
				continue
			}
			data := trimmed[5..].trim_space()
			if data == '[DONE]' {
				break
			}
			chunk := json.decode(ChatCompletionChunk, data) or { continue }
			if chunk.choices.len > 0 {
				content := chunk.choices[0].delta.content
				if content != '' {
					builder.write_string(content)
				}
			}
			if chunk.usage.total_tokens > 0 {
				usage = chunk.usage
			}
		}
		content := builder.str()
		return ModelResult{
			content: content
			prompt_tokens: usage.prompt_tokens
			completion_tokens: usage.completion_tokens
			total_tokens: usage.total_tokens
		}
	}
	full := json.decode(ChatCompletionResponse, body)!
	mut content := ''
	if full.choices.len > 0 {
		content = full.choices[0].message.content
	}
	return ModelResult{
		content: content
		prompt_tokens: full.usage.prompt_tokens
		completion_tokens: full.usage.completion_tokens
		total_tokens: full.usage.total_tokens
	}
}

fn call_embedding(cfg Config, text string) ![]f64 {
	if cfg.embedding_model.trim_space() == '' {
		return error('embedding model is not configured')
	}
	payload := EmbeddingRequest{
		model: cfg.embedding_model
		input: text
	}
	mut req := http.Request{
		method: .post
		url: cfg.base_url + '/embeddings'
		data: json.encode(payload)
	}
	req.add_header(.content_type, 'application/json')
	req.add_header(.authorization, 'Bearer ' + cfg.api_key)
	resp := req.do()!
	if resp.status_code < 200 || resp.status_code >= 300 {
		return error('embedding api error ${resp.status_code}: ' + truncate_text(resp.body, 2000))
	}
	parsed := json.decode(EmbeddingResponse, resp.body)!
	if parsed.data.len == 0 {
		return error('embedding response contained no vectors')
	}
	return parsed.data[0].embedding
}

fn encode_vector(vec []f64) string {
	mut arr := []JsonNode{}
	for value in vec {
		arr << jnum(value)
	}
	return serialize_json_node(jarr(arr))
}

fn decode_vector(raw string) ![]f64 {
	if raw.trim_space() == '' {
		return []f64{}
	}
	node := parse_json_node(raw)!
	if node.kind != .array_value {
		return error('vector json is not an array')
	}
	mut values := []f64{}
	for item in node.arr {
		values << node_as_f64(item)!
	}
	return values
}

fn cosine(a []f64, b []f64) f64 {
	mut n := a.len
	if b.len < n {
		n = b.len
	}
	if n == 0 {
		return 0.0
	}
	mut dot := 0.0
	mut norm_a := 0.0
	mut norm_b := 0.0
	for i := 0; i < n; i++ {
		dot += a[i] * b[i]
		norm_a += a[i] * a[i]
		norm_b += b[i] * b[i]
	}
	if norm_a <= 0.0 || norm_b <= 0.0 {
		return 0.0
	}
	return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
}

fn fts_query_from_text(text string) string {
	mut terms := []string{}
	mut current := strings.new_builder(16)
	for i := 0; i < text.len; i++ {
		c := text[i]
		if is_alnum(c) {
			current.write_u8(c)
		} else {
			word := current.str().to_lower()
			if word.len >= 3 && terms.len < 16 {
				terms << word
			}
			current = strings.new_builder(16)
		}
	}
	word := current.str().to_lower()
	if word.len >= 3 && terms.len < 16 {
		terms << word
	}
	if terms.len == 0 {
		return ''
	}
	return terms.join(' OR ')
}

fn add_rrf(mut scores map[string]f64, ranked []string, k f64) {
	for index, id in ranked {
		mut current := 0.0
		if id in scores {
			current = scores[id]
		}
		scores[id] = current + (1.0 / (k + f64(index + 1)))
	}
}

fn compact_state_query(state_json string, wm_json string) string {
	return truncate_text(state_json + '\n' + wm_json, 6000)
}

fn retrieve_skills(cfg Config, tenant_id string, state_json string, wm_json string) ![]Skill {
	query_text := compact_state_query(state_json, wm_json)
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	mut scores := map[string]f64{}
	fts_query := fts_query_from_text(query_text)
	if fts_query != '' {
		fts_rows := db.exec_param_many('select id, bm25(skills_fts) from skills_fts where skills_fts match ? and (tenant_id = ? or tenant_id = ?) limit 20', [
			fts_query,
			tenant_id,
			'global',
		]) or { []sqlite.Row{} }
		mut ranked := []string{}
		for row in fts_rows {
			id := row_get(row, 0)
			if id != '' {
				ranked << id
			}
		}
		add_rrf(mut scores, ranked, 60.0)
	}
	query_embedding := call_embedding(cfg, query_text) or { []f64{} }
	if query_embedding.len > 0 {
		rows := db.exec_param_many('select id, tenant_id, name, description, trigger_json, body, embedding_json, version, enabled, tests_json, updated_at_ms from skills where enabled = ? and (tenant_id = ? or tenant_id = ?)', [
			'1',
			tenant_id,
			'global',
		])!
		mut dense_scores := []DenseScore{}
		for row in rows {
			skill := skill_from_row(row)
			vec := decode_vector(skill.embedding_json) or { []f64{} }
			if vec.len == 0 {
				continue
			}
			dense_scores << DenseScore{
				id: skill.id
				score: cosine(query_embedding, vec)
			}
		}
		dense_scores.sort(a.score > b.score)
		mut dense_ranked := []string{}
		for index, ds in dense_scores {
			if index >= 20 {
				break
			}
			dense_ranked << ds.id
		}
		add_rrf(mut scores, dense_ranked, 60.0)
	}
	mut fused := []ScoredId{}
	for id, score in scores {
		fused << ScoredId{
			id: id
			score: score
		}
	}
	fused.sort(a.score > b.score)
	mut ids := []string{}
	for index, scored in fused {
		if index >= 2 {
			break
		}
		ids << scored.id
	}
	if ids.len == 0 {
		rows := db.exec_param_many('select id, tenant_id, name, description, trigger_json, body, embedding_json, version, enabled, tests_json, updated_at_ms from skills where enabled = ? and (tenant_id = ? or tenant_id = ?) order by updated_at_ms desc limit 2', [
			'1',
			tenant_id,
			'global',
		])!
		mut fallback := []Skill{}
		for row in rows {
			fallback << skill_from_row(row)
		}
		return fallback
	}
	mut result := []Skill{}
	for id in ids {
		rows := db.exec_param_many('select id, tenant_id, name, description, trigger_json, body, embedding_json, version, enabled, tests_json, updated_at_ms from skills where id = ? and enabled = ? and (tenant_id = ? or tenant_id = ?) order by updated_at_ms desc limit 1', [
			id,
			'1',
			tenant_id,
			'global',
		])!
		if rows.len > 0 {
			result << skill_from_row(rows[0])
		}
	}
	return result
}

fn skills_to_prompt_node(skills []Skill) JsonNode {
	mut arr := []JsonNode{}
	for skill in skills {
		arr << jobj({
			'id': jstr(skill.id)
			'name': jstr(skill.name)
			'description': jstr(skill.description)
			'version': jnum(f64(skill.version))
			'procedure': jstr(truncate_text(skill.body, 4000))
			'trigger': parse_json_node(skill.trigger_json) or { jobj(map[string]JsonNode{}) }
		})
	}
	return jarr(arr)
}

fn load_recent_playbook_lessons(cfg Config, tenant_id string) string {
	root := os.join_path(cfg.knowledge_root, tenant_id)
	path := os.join_path(root, 'playbook.md')
	if !os.exists(path) {
		return 'none'
	}
	content := os.read_file(path) or { return 'none' }
	lines := content.split_into_lines()
	if lines.len == 0 {
		return 'none'
	}
	start := if lines.len > 8 { lines.len - 8 } else { 0 }
	return truncate_text(lines[start..].join('\n'), 3000)
}

fn load_policy_signals(mut db sqlite.DB, tenant_id string) string {
	rows := db.exec_param_many('select key, weight from policy_weights where tenant_id = ? order by weight desc limit 10', [
		tenant_id,
	]) or { return '[]' }
	if rows.len == 0 {
		return '[]'
	}
	mut items := []JsonNode{}
	for row in rows {
		k := row_get(row, 0)
		w := strconv.atof64(row_get(row, 1)) or { 0.0 }
		items << jobj({
			'pattern': jstr(k)
			'weight': jnum(w)
		})
	}
	return serialize_json_node(jarr(items))
}

fn build_step_messages(cfg Config, tenant_id string, spec_json string, state_json string, wm_json string, observation_json string, skills []Skill, lessons string, policy string) []ChatMessage {
	system := 'You are a frozen stateless language model used only as a deterministic state-transition proposer. Think privately if needed, but output exactly one JSON object and no markdown. Never output hidden reasoning, scratchpads, dialogue transcripts, or prior actions. The runtime prompt contains only the immutable procedural specification, active knowledge lessons, policy signals, verified structured execution state, latest observation, working memory, and experiential skills. Produce a bounded state patch, a bounded working memory patch, one action command, a terminal flag, and cognition metadata. Null values inside patches delete keys. All keys must be stable structured state keys.'
	schema := '{"state_patch":{"status":"running"},"working_memory_patch":{"open_goals":["remaining goal"]},"action":{"tool":"noop","arguments":{},"idempotency_key":"stable-key","requires_authorization":true},"terminal":false,"selected_skill_id":"","cognition":{"transition_gate":0.0,"subgoals":[],"tokens":[[0.0]]}}'
	user := 'PROCEDURAL_SPECIFICATION_P:\n' + spec_json + '\n\nHISTORICAL_FAILURE_LESSONS_AND_WORKAROUNDS:\n' +
		lessons + '\n\nPOLICY_DISTILLATION_PRIORS:\n' + policy +
		'\n\nSTRUCTURED_EXECUTION_STATE_SIGMA_T:\n' + state_json +
		'\n\nWORKING_MEMORY_WM:\n' + wm_json + '\n\nLATEST_OBSERVATION_O_T:\n' +
		observation_json + '\n\nRELEVANT_EXPERIENTIAL_SKILLS_EM:\n' +
		serialize_json_node(skills_to_prompt_node(skills)) +
		'\n\nOUTPUT_SCHEMA_EXAMPLE:\n' + schema +
		'\n\nReturn exactly one JSON object matching the schema. Do not include history or reasoning.'
	return [
		ChatMessage{
			role: 'system'
			content: system
		},
		ChatMessage{
			role: 'user'
			content: user
		},
	]
}

fn forbidden_state_key(key string) bool {
	lower := key.to_lower()
	return lower.contains('transcript') || lower.contains('conversation_history') || lower == 'history'
		|| lower.contains('chain_of_thought') || lower.contains('scratchpad') || lower.contains('reasoning_trace')
}

fn validate_state_key(key string) ! {
	if key.len == 0 || key.len > 96 {
		return error('invalid state key length')
	}
	if forbidden_state_key(key) {
		return error('state key is forbidden: ' + key)
	}
	for i := 0; i < key.len; i++ {
		c := key[i]
		if !(is_alnum(c) || c == `_` || c == `-` || c == `.` || c == `:`) {
			return error('invalid state key character')
		}
	}
}

fn validate_patch_node(node JsonNode, depth int) !int {
	if depth > 12 {
		return error('patch depth exceeds limit')
	}
	match node.kind {
		.null_value, .bool_value, .number_value {
			return 1
		}
		.string_value {
			if node.s.len > 8192 {
				return error('patch string value too large')
			}
			return 1
		}
		.array_value {
			if node.arr.len > 128 {
				return error('patch array too large')
			}
			mut total := 1
			for item in node.arr {
				total += validate_patch_node(item, depth + 1)!
				if total > 2048 {
					return error('patch node count exceeds limit')
				}
			}
			return total
		}
		.object_value {
			if node.obj.len > 128 {
				return error('patch object too large')
			}
			mut total := 1
			for key, value in node.obj {
				validate_state_key(key)!
				total += validate_patch_node(value, depth + 1)!
				if total > 2048 {
					return error('patch count exceeds limit')
				}
			}
			return total
		}
	}
}

fn merge_patch(current JsonNode, patch JsonNode, depth int) !JsonNode {
	if current.kind != .object_value {
		return error('current state is not an object')
	}
	if patch.kind != .object_value {
		return error('patch is not an object')
	}
	if depth > 12 {
		return error('merge depth exceeds limit')
	}
	mut out := current.obj.clone()
	for key, patch_value in patch.obj {
		validate_state_key(key)!
		if patch_value.kind == .null_value {
			out.delete(key)
			continue
		}
		if key in out && out[key].kind == .object_value && patch_value.kind == .object_value {
			out[key] = merge_patch(out[key], patch_value, depth + 1)!
		} else {
			validate_patch_node(patch_value, depth + 1)!
			out[key] = patch_value
		}
	}
	return jobj(out)
}

fn apply_patch_json(current_json string, patch JsonNode, max_bytes int) !string {
	current := parse_json_node(current_json)!
	if current.kind != .object_value {
		return error('current json is not an object')
	}
	validate_patch_node(patch, 0)!
	merged := merge_patch(current, patch, 0)!
	serialized := serialize_json_node(merged)
	if serialized.len > max_bytes {
		return error('merged state exceeds byte limit')
	}
	return serialized
}

fn sanitize_decision_node(node JsonNode) JsonNode {
	if node.kind != .object_value {
		return node
	}
	mut obj := node.obj.clone()
	obj.delete('reasoning')
	obj.delete('rationale')
	obj.delete('chain_of_thought')
	obj.delete('scratchpad')
	obj.delete('transcript')
	obj.delete('history')
	return jobj(obj)
}

fn parse_cognition_tokens(node JsonNode) ![][]f64 {
	if node.kind != .array_value {
		return [][]f64{}
	}
	if node.arr.len > 8 {
		return error('too many cognition token rows')
	}
	mut rows := [][]f64{}
	for row_node in node.arr {
		if row_node.kind != .array_value {
			return error('cognition row is not array')
		}
		if row_node.arr.len > 64 {
			return error('cognition row too wide')
		}
		mut row := []f64{}
		for item in row_node.arr {
			row << node_as_f64(item)!
		}
		rows << row
	}
	return rows
}

fn decode_step_decision(raw string) !StepDecision {
	json_text := extract_first_json_object(raw)!
	root_raw := parse_json_node(json_text)!
	if root_raw.kind != .object_value {
		return error('decision root is not object')
	}
	root := sanitize_decision_node(root_raw)
	state_patch := node_get(root, 'state_patch') or { return error('missing state_patch') }
	if state_patch.kind != .object_value {
		return error('state_patch must be object')
	}
	wm_patch := node_get(root, 'working_memory_patch') or { jobj(map[string]JsonNode{}) }
	if wm_patch.kind != .object_value {
		return error('working_memory_patch must be object')
	}
	action_node := node_get(root, 'action') or { return error('missing action') }
	if action_node.kind != .object_value {
		return error('action must be object')
	}
	tool := required_string_field(action_node, 'tool')!
	arguments := node_get(action_node, 'arguments') or { jobj(map[string]JsonNode{}) }
	if arguments.kind != .object_value {
		return error('action arguments must be object')
	}
	idempotency_key := optional_string_field(action_node, 'idempotency_key', sha_hex(tool + serialize_json_node(arguments))[..32])
	requires_authorization := optional_bool_field(action_node, 'requires_authorization', true)
	cognition_node := node_get(root, 'cognition') or { jobj(map[string]JsonNode{}) }
	mut transition_gate := 0.0
	mut cognition_tokens := [][]f64{}
	mut subgoals := []string{}
	if cognition_node.kind == .object_value {
		transition_gate = optional_f64_field(cognition_node, 'transition_gate', 0.0)
		tokens_node := node_get(cognition_node, 'tokens') or { jarr([]JsonNode{}) }
		cognition_tokens = parse_cognition_tokens(tokens_node)!
		subgoal_node := node_get(cognition_node, 'subgoals') or { jarr([]JsonNode{}) }
		if subgoal_node.kind == .array_value {
			subgoals = node_string_array(subgoal_node) or { []string{} }
		}
	}
	top_subgoal_node := node_get(root, 'subgoals') or { jarr([]JsonNode{}) }
	if top_subgoal_node.kind == .array_value && subgoals.len == 0 {
		subgoals = node_string_array(top_subgoal_node) or { []string{} }
	}
	return StepDecision{
		state_patch: state_patch
		wm_patch: wm_patch
		action: ActionCommand{
			tool: tool
			arguments: arguments
			idempotency_key: idempotency_key
			requires_authorization: requires_authorization
		}
		terminal: optional_bool_field(root, 'terminal', false)
		selected_skill_id: optional_string_field(root, 'selected_skill_id', '')
		cognition_tokens: cognition_tokens
		transition_gate: transition_gate
		subgoals: subgoals
		schema_json: serialize_json_node(root)
	}
}

fn action_command_to_node(cmd ActionCommand, selected_skill_id string, terminal bool) JsonNode {
	return jobj({
		'tool': jstr(cmd.tool)
		'arguments': cmd.arguments
		'idempotency_key': jstr(cmd.idempotency_key)
		'requires_authorization': jbool(cmd.requires_authorization)
		'selected_skill_id': jstr(selected_skill_id)
		'terminal': jbool(terminal)
	})
}

fn action_command_from_node(node JsonNode) !ActionCommand {
	if node.kind != .object_value {
		return error('action row json is not object')
	}
	tool := required_string_field(node, 'tool')!
	args := node_get(node, 'arguments') or { jobj(map[string]JsonNode{}) }
	if args.kind != .object_value {
		return error('action arguments are not object')
	}
	return ActionCommand{
		tool: tool
		arguments: args
		idempotency_key: optional_string_field(node, 'idempotency_key', sha_hex(tool + serialize_json_node(args))[..32])
		requires_authorization: optional_bool_field(node, 'requires_authorization', true)
	}
}

fn cognition_to_json(decision StepDecision) string {
	mut token_rows := []JsonNode{}
	for row in decision.cognition_tokens {
		mut values := []JsonNode{}
		for value in row {
			values << jnum(value)
		}
		token_rows << jarr(values)
	}
	return serialize_json_node(jobj({
		'tokens': jarr(token_rows)
		'transition_gate': jnum(decision.transition_gate)
		'subgoals': string_array_node(decision.subgoals)
		'generated_at_ms': jnum(f64(now_ms()))
	}))
}

fn validate_tool_schema(session SessionRecord, cmd ActionCommand) ! {
	match cmd.tool {
		'noop' {}
		'final_answer' {
			answer := required_string_field(cmd.arguments, 'answer')!
			if answer.trim_space() == '' {
				return error('final_answer requires nonempty answer')
			}
			if answer.len > 65536 {
				return error('final_answer too large')
			}
		}
		'append_file' {
			path := required_string_field(cmd.arguments, 'path')!
			safe_join(session.workspace_path, path)!
			has_content := (node_get(cmd.arguments, 'content') or { jnull() }).kind == .string_value
			has_lines := (node_get(cmd.arguments, 'lines') or { jnull() }).kind == .array_value
			if !has_content && !has_lines {
				return error('append_file requires content or lines')
			}
		}
		'replace_lines' {
			path := required_string_field(cmd.arguments, 'path')!
			safe_join(session.workspace_path, path)!
			start_line := optional_int_field(cmd.arguments, 'start_line', 0)
			end_line := optional_int_field(cmd.arguments, 'end_line', 0)
			if start_line < 1 || end_line < start_line {
				return error('replace_lines requires valid one-based inclusive range')
			}
		}
		'check_lines' {
			path := required_string_field(cmd.arguments, 'path')!
			safe_join(session.workspace_path, path)!
			lines_node := node_get(cmd.arguments, 'lines') or { return error('check_lines requires lines') }
			_ := node_string_array(lines_node)!
		}
		'read_file' {
			path := required_string_field(cmd.arguments, 'path')!
			safe_join(session.workspace_path, path)!
		}
		'list_files' {
			path := optional_string_field(cmd.arguments, 'path', '')
			safe_join(session.workspace_path, path)!
		}
		else {
			return error('unknown or unauthorized tool name: ' + cmd.tool)
		}
	}
}

fn validate_transition(cfg Config, session SessionRecord, decision StepDecision) !ValidatedTransition {
	new_state_json := apply_patch_json(session.state_json, decision.state_patch, cfg.max_state_bytes)!
	new_wm_json := apply_patch_json(session.wm_json, decision.wm_patch, cfg.max_state_bytes)!
	validate_tool_schema(session, decision.action)!
	action_json := serialize_json_node(action_command_to_node(decision.action, decision.selected_skill_id, decision.terminal))
	cognition_json := cognition_to_json(decision)
	return ValidatedTransition{
		new_state_json: new_state_json
		new_wm_json: new_wm_json
		action_json: action_json
		cognition_json: cognition_json
	}
}

fn validation_error_observation(message string) string {
	return serialize_json_node(jobj({
		'type': jstr('validation_error')
		'message': jstr(truncate_text(message, 8000))
		'at_ms': jnum(f64(now_ms()))
	}))
}

fn release_session_deliberation_lease(cfg Config, tenant_id string, session_id string) {
	mut db := open_db(cfg) or { return }
	defer {
		db.close() or {}
	}
	db.exec_param_many('update sessions set lease_expires_ms = 0 where tenant_id = ? and id = ?', [
		tenant_id,
		session_id,
	]) or {}
}

fn run_deliberative_step(cfg Config, tenant_id string, session_id string) ! {
	defer {
		release_session_deliberation_lease(cfg, tenant_id, session_id)
	}
	mut attempts := 0
	mut retry_observation := ''
	for attempts < cfg.step_retry_limit {
		attempts++
		mut db_read := open_db(cfg)!
		session := load_session(mut db_read, tenant_id, session_id)!
		if session.status != 'running' {
			db_read.close() or {}
			return
		}
		if session.last_step >= session.max_steps {
			db_read.close() or {}
			mark_session_failed(cfg, tenant_id, session_id, 'maximum step count reached')!
			if cfg.enable_learning {
				post_trajectory_learning(cfg, tenant_id, session_id) or { eprintln('learning error: ${err}') }
			}
			return
		}
		pending := pending_action_count(mut db_read, tenant_id, session_id)!
		lessons := load_recent_playbook_lessons(cfg, tenant_id)
		policy := load_policy_signals(mut db_read, tenant_id)
		db_read.close() or {}
		if pending > 0 {
			return
		}
		obs_json := if retry_observation != '' { retry_observation } else { session.observation_json }
		skills := retrieve_skills(cfg, tenant_id, session.state_json, session.wm_json)!
		messages := build_step_messages(cfg, tenant_id, session.spec_json, session.state_json, session.wm_json, obs_json,
			skills, lessons, policy)
		prompt_size := messages[0].content.len + messages[1].content.len
		if session.tokens_used + estimate_tokens(messages[1].content) + cfg.max_model_tokens > session.token_budget {
			mark_session_failed(cfg, tenant_id, session_id, 'token budget exhausted before model call')!
			if cfg.enable_learning {
				post_trajectory_learning(cfg, tenant_id, session_id) or { eprintln('learning error: ${err}') }
			}
			return
		}
		temperature := if attempts == 1 { 0.7 } else if attempts == 2 { 0.1 } else { 0.0 }
		freq_penalty := if attempts == 1 { 0.2 } else { 0.0 }
		pres_penalty := if attempts == 1 { 0.2 } else { 0.0 }
		model_result := call_model_stream(cfg, messages, temperature, cfg.max_model_tokens, freq_penalty, pres_penalty) or {
			retry_observation = validation_error_observation('model call failed: ${err}')
			continue
		}
		decision := decode_step_decision(model_result.content) or {
			retry_observation = validation_error_observation('schema decode failed: ${err}')
			continue
		}
		transition := validate_transition(cfg, session, decision) or {
			retry_observation = validation_error_observation('transition validation failed: ${err}')
			continue
		}
		step := session.last_step + 1
		mut commit_ok := false
		{
			mut db := open_db(cfg)!
			defer {
				db.close() or {}
			}
			db.exec('begin immediate transaction') or {
				retry_observation = validation_error_observation('database transaction start failed: ${err}')
				continue
			}
			fresh := load_session(mut db, tenant_id, session_id) or {
				db.exec('rollback') or {}
				retry_observation = validation_error_observation('session reloading failed')
				continue
			}
			if fresh.last_step != session.last_step || fresh.state_json != session.state_json || fresh.status != 'running' {
				db.exec('rollback') or {}
				retry_observation = validation_error_observation('checkpoint changed concurrently')
				continue
			}
			action_id := new_id('action')
			db.exec_param_many('insert into actions(id, tenant_id, session_id, step, action_json, idempotency_key, status, result_json, error, created_at_ms, updated_at_ms) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
				action_id,
				tenant_id,
				session_id,
				step.str(),
				transition.action_json,
				decision.action.idempotency_key,
				'pending',
				'{}',
				'',
				now_ms().str(),
				now_ms().str(),
			]) or {
				db.exec('rollback') or {}
				retry_observation = validation_error_observation('action creation failed')
				continue
			}
			new_tokens_used := session.tokens_used + if model_result.total_tokens > 0 {
				model_result.total_tokens
			} else {
				estimate_tokens(model_result.content) + estimate_tokens(messages[0].content) + estimate_tokens(messages[1].content)
			}
			db.exec_param_many('update sessions set state_json = ?, wm_json = ?, last_step = ?, tokens_used = ?, lease_expires_ms = 0, updated_at_ms = ? where tenant_id = ? and id = ?', [
				transition.new_state_json,
				transition.new_wm_json,
				step.str(),
				new_tokens_used.str(),
				now_ms().str(),
				tenant_id,
				session_id,
			]) or {
				db.exec('rollback') or {}
				retry_observation = validation_error_observation('session update failed')
				continue
			}
			insert_checkpoint(mut db, tenant_id, session_id, step, 'system2', transition.new_state_json,
				transition.new_wm_json, obs_json, serialize_json_node(decision.state_patch), transition.action_json) or {
				db.exec('rollback') or {}
				retry_observation = validation_error_observation('checkpoint insert failed')
				continue
			}
			db.exec_param_many('insert into cognition_frames(id, tenant_id, session_id, step, cognition_json, generated_at_ms) values(?, ?, ?, ?, ?, ?)', [
				new_id('cognition'),
				tenant_id,
				session_id,
				step.str(),
				transition.cognition_json,
				now_ms().str(),
			]) or {
				db.exec('rollback') or {}
				retry_observation = validation_error_observation('cognition frame failed')
				continue
			}
			insert_raw_trace(mut db, tenant_id, session_id, jobj({
				'type': jstr('verified_system2_transition')
				'step': jnum(f64(step))
				'prompt_hash': jstr(sha_hex(session.spec_json + session.state_json + session.wm_json + obs_json + serialize_json_node(skills_to_prompt_node(skills))))
				'prompt_bytes': jnum(f64(prompt_size))
				'state_before_hash': jstr(sha_hex(session.state_json))
				'state_after_hash': jstr(sha_hex(transition.new_state_json))
				'working_memory_after_hash': jstr(sha_hex(transition.new_wm_json))
				'selected_skill_id': jstr(decision.selected_skill_id)
				'action': parse_json_node(transition.action_json) or { jobj(map[string]JsonNode{}) }
				'model_usage': jobj({
					'prompt_tokens': jnum(f64(model_result.prompt_tokens))
					'completion_tokens': jnum(f64(model_result.completion_tokens))
					'total_tokens': jnum(f64(model_result.total_tokens))
				})
			})) or {
				db.exec('rollback') or {}
				retry_observation = validation_error_observation('trace recording failed')
				continue
			}
			db.exec('commit') or {
				retry_observation = validation_error_observation('commit failed')
				continue
			}
			commit_ok = true
		}
		if commit_ok {
			return
		}
	}
	mark_session_failed(cfg, tenant_id, session_id, 'all retry attempts failed; latest observation: ' + retry_observation)!
	if cfg.enable_learning {
		post_trajectory_learning(cfg, tenant_id, session_id) or { eprintln('learning error: ${err}') }
	}
}

fn try_claim_session_for_deliberation(cfg Config, tenant_id string, session_id string) bool {
	mut db := open_db(cfg) or { return false }
	defer {
		db.close() or {}
	}
	db.exec('begin immediate transaction') or { return false }
	now := now_ms()
	rows := db.exec_param_many('select lease_expires_ms from sessions where tenant_id = ? and id = ? and status = ?', [
		tenant_id,
		session_id,
		'running',
	]) or {
		db.exec('rollback') or {}
		return false
	}
	if rows.len == 0 {
		db.exec('rollback') or {}
		return false
	}
	lease := parse_i64_default(row_get(rows[0], 0), 0)
	if lease > now {
		db.exec('rollback') or {}
		return false
	}
	pending_rows := db.exec_param_many('select count(*) from actions where tenant_id = ? and session_id = ? and (status = ? or status = ?)', [
		tenant_id,
		session_id,
		'pending',
		'running',
	]) or {
		db.exec('rollback') or {}
		return false
	}
	if pending_rows.len > 0 && parse_int_default(row_get(pending_rows[0], 0), 0) > 0 {
		db.exec('rollback') or {}
		return false
	}
	new_lease := now + cfg.deliberation_lease_ms
	db.exec_param_many('update sessions set lease_expires_ms = ?, updated_at_ms = ? where tenant_id = ? and id = ? and status = ?', [
		new_lease.str(),
		now.str(),
		tenant_id,
		session_id,
		'running',
	]) or {
		db.exec('rollback') or {}
		return false
	}
	db.exec('commit') or { return false }
	return true
}

fn recover_stale_sessions(mut db sqlite.DB) ! {
	cutoff := (now_ms() - 180000).str()
	db.exec_param_many('update sessions set status = ?, error = ?, lease_expires_ms = 0, updated_at_ms = ? where status = ? and updated_at_ms < ? and lease_expires_ms < ?', [
		'failed',
		'session lease timed out',
		now_ms().str(),
		'running',
		cutoff,
		now_ms().str(),
	])!
}

fn process_system2_tick(cfg Config) ! {
	mut db := open_db(cfg)!
	recover_stale_sessions(mut db) or {}
	now := now_ms().str()
	rows := db.exec_param_many('select tenant_id, id from sessions where status = ? and last_step < max_steps and lease_expires_ms <= ? order by updated_at_ms asc limit 16', [
		'running',
		now,
	])!
	db.close() or {}
	for row in rows {
		tenant_id := row_get(row, 0)
		session_id := row_get(row, 1)
		if try_claim_session_for_deliberation(cfg, tenant_id, session_id) {
			spawn fn (c Config, tid string, sid string) {
				run_deliberative_step(c, tid, sid) or {
					eprintln('system2 step error: ${err}')
				}
			}(cfg, tenant_id, session_id)
		}
	}
}

fn system2_loop(cfg Config) {
	for {
		process_system2_tick(cfg) or { eprintln('system2 tick error: ${err}') }
		time.sleep(1 * time.second)
	}
}

fn safe_join(root string, rel string) !string {
	mut cleaned := rel.replace('\\', '/').trim_space()
	if cleaned == '' {
		return root
	}
	if cleaned.starts_with('/') || cleaned.contains(':') || cleaned.contains('\x00') {
		return error('unsafe path')
	}
	mut path := root
	for part in cleaned.split('/') {
		if part == '' || part == '.' {
			continue
		}
		if part == '..' {
			return error('path traversal rejected')
		}
		for i := 0; i < part.len; i++ {
			c := part[i]
			if c < 32 {
				return error('invalid path control character')
			}
		}
		path = os.join_path(path, part)
	}
	return path
}

fn atomic_write_file(path string, content string) ! {
	dir := os.dir(path)
	if dir != '' {
		os.mkdir_all(dir)!
	}
	tmp := path + '.tmp.' + new_id('atomic')
	os.write_file(tmp, content)!
	os.rename(tmp, path)!
}

fn file_hash(path string) !string {
	if !os.exists(path) {
		return ''
	}
	content := os.read_file(path)!
	return sha_hex(content)
}

fn lines_from_arguments(args JsonNode) ![]string {
	lines_node := node_get(args, 'lines') or { jnull() }
	if lines_node.kind == .array_value {
		return node_string_array(lines_node)
	}
	content := required_string_field(args, 'content')!
	if content.contains('\n') {
		return content.split_into_lines()
	}
	return [content]
}

fn append_lines_file(path string, lines []string, unique bool) !JsonNode {
	mut existing := []string{}
	if os.exists(path) {
		existing = os.read_file(path)!.split_into_lines()
	}
	mut seen := map[string]bool{}
	for line in existing {
		seen[line] = true
	}
	mut final_lines := existing.clone()
	mut appended := []string{}
	for line in lines {
		if unique && line in seen {
			continue
		}
		final_lines << line
		seen[line] = true
		appended << line
	}
	mut content := final_lines.join('\n')
	if content.len > 0 {
		content += '\n'
	}
	atomic_write_file(path, content)!
	return jobj({
		'appended_count': jnum(f64(appended.len))
		'total_lines': jnum(f64(final_lines.len))
		'appended_lines': string_array_node(appended)
	})
}

fn replace_lines_file(path string, start_line int, end_line int, replacement []string) !JsonNode {
	if start_line < 1 || end_line < start_line {
		return error('invalid line range')
	}
	mut existing := []string{}
	if os.exists(path) {
		existing = os.read_file(path)!.split_into_lines()
	}
	start_index := start_line - 1
	end_exclusive := end_line
	if start_index > existing.len || end_exclusive > existing.len {
		return error('line range outside file')
	}
	mut out := []string{}
	for i := 0; i < start_index; i++ {
		out << existing[i]
	}
	for line in replacement {
		out << line
	}
	for i := end_exclusive; i < existing.len; i++ {
		out << existing[i]
	}
	mut content := out.join('\n')
	if content.len > 0 {
		content += '\n'
	}
	atomic_write_file(path, content)!
	return jobj({
		'start_line': jnum(f64(start_line))
		'end_line': jnum(f64(end_line))
		'replacement_count': jnum(f64(replacement.len))
		'total_lines': jnum(f64(out.len))
	})
}

fn check_lines_exact(path string, expected []string) !map[string]bool {
	mut existing := map[string]bool{}
	if os.exists(path) {
		for line in os.read_file(path)!.split_into_lines() {
			existing[line] = true
		}
	}
	mut result := map[string]bool{}
	for line in expected {
		result[line] = line in existing
	}
	return result
}

fn check_lines_node(values map[string]bool) JsonNode {
	mut obj := map[string]JsonNode{}
	for key, value in values {
		obj[key] = jbool(value)
	}
	return jobj(obj)
}

fn list_files_recursive(root string, rel string, max_count int, mut out []string) ! {
	if out.len >= max_count {
		return
	}
	dir := safe_join(root, rel)!
	if !os.exists(dir) {
		return
	}
	entries := os.ls(dir)!
	mut sorted := entries.clone()
	sorted.sort()
	for entry in sorted {
		if out.len >= max_count {
			return
		}
		next_rel := if rel == '' { entry } else { rel + '/' + entry }
		full := safe_join(root, next_rel)!
		if os.is_dir(full) {
			list_files_recursive(root, next_rel, max_count, mut out)!
		} else {
			out << next_rel
		}
	}
}

fn authorize_tool(cfg Config, mut db sqlite.DB, tenant_id string, session_id string, cmd ActionCommand) !bool {
	match cmd.tool {
		'noop', 'final_answer', 'read_file', 'list_files', 'check_lines' {
			return true
		}
		'append_file', 'replace_lines' {
			if cfg.allow_write_tools {
				return true
			}
			path := required_string_field(cmd.arguments, 'path')!
			rows := db.exec_param_many('select path_prefix from tool_authorizations where tenant_id = ? and session_id = ? and tool_name = ? and expires_at_ms > ? order by expires_at_ms desc', [
				tenant_id,
				session_id,
				cmd.tool,
				now_ms().str(),
			])!
			for row in rows {
				prefix := row_get(row, 0)
				if prefix == '' || path.starts_with(prefix) {
					return true
				}
			}
			return false
		}
		else {
			return false
		}
	}
}

fn unauthorized_tool_result(cmd ActionCommand) ToolExecution {
	obs := jobj({
		'type': jstr('tool_authorization_denied')
		'tool': jstr(cmd.tool)
		'ok': jbool(false)
		'at_ms': jnum(f64(now_ms()))
	})
	return ToolExecution{
		ok: false
		observation: obs
		receipt: obs
		terminal: false
		final_answer: ''
	}
}

fn bound_observation(node JsonNode, max_bytes int) JsonNode {
	serialized := serialize_json_node(node)
	if serialized.len <= max_bytes {
		return node
	}
	return jobj({
		'type': jstr('truncated_observation')
		'truncated': jbool(true)
		'content': jstr(truncate_text(serialized, max_bytes))
		'at_ms': jnum(f64(now_ms()))
	})
}

fn execute_tool(cfg Config, cmd ActionCommand, workspace string) !ToolExecution {
	match cmd.tool {
		'noop' {
			obs := jobj({
				'type': jstr('noop')
				'ok': jbool(true)
				'at_ms': jnum(f64(now_ms()))
			})
			return ToolExecution{
				ok: true
				observation: obs
				receipt: obs
				terminal: false
				final_answer: ''
			}
		}
		'final_answer' {
			answer := required_string_field(cmd.arguments, 'answer')!
			obs := bound_observation(jobj({
				'type': jstr('final_answer')
				'ok': jbool(true)
				'answer': jstr(answer)
				'at_ms': jnum(f64(now_ms()))
			}), cfg.max_observation_bytes)
			return ToolExecution{
				ok: true
				observation: obs
				receipt: obs
				terminal: true
				final_answer: answer
			}
		}
		'append_file' {
			rel := required_string_field(cmd.arguments, 'path')!
			full := safe_join(workspace, rel)!
			before := file_hash(full) or { '' }
			lines := lines_from_arguments(cmd.arguments)!
			unique := optional_bool_field(cmd.arguments, 'unique', false)
			result := append_lines_file(full, lines, unique)!
			after := file_hash(full)!
			receipt := jobj({
				'type': jstr('append_file_receipt')
				'ok': jbool(true)
				'path': jstr(rel)
				'before_hash': jstr(before)
				'after_hash': jstr(after)
				'result': result
				'at_ms': jnum(f64(now_ms()))
			})
			return ToolExecution{
				ok: true
				observation: bound_observation(receipt, cfg.max_observation_bytes)
				receipt: receipt
				terminal: false
				final_answer: ''
			}
		}
		'replace_lines' {
			rel := required_string_field(cmd.arguments, 'path')!
			full := safe_join(workspace, rel)!
			before := file_hash(full) or { '' }
			start_line := optional_int_field(cmd.arguments, 'start_line', 0)
			end_line := optional_int_field(cmd.arguments, 'end_line', 0)
			replacement := lines_from_arguments(cmd.arguments)!
			result := replace_lines_file(full, start_line, end_line, replacement)!
			after := file_hash(full)!
			receipt := jobj({
				'type': jstr('replace_lines_receipt')
				'ok': jbool(true)
				'path': jstr(rel)
				'before_hash': jstr(before)
				'after_hash': jstr(after)
				'result': result
				'at_ms': jnum(f64(now_ms()))
			})
			return ToolExecution{
				ok: true
				observation: bound_observation(receipt, cfg.max_observation_bytes)
				receipt: receipt
				terminal: false
				final_answer: ''
			}
		}
		'check_lines' {
			rel := required_string_field(cmd.arguments, 'path')!
			full := safe_join(workspace, rel)!
			lines_node := node_get(cmd.arguments, 'lines') or { return error('missing lines') }
			lines := node_string_array(lines_node)!
			result := check_lines_exact(full, lines)!
			receipt := jobj({
				'type': jstr('check_lines_receipt')
				'ok': jbool(true)
				'path': jstr(rel)
				'results': check_lines_node(result)
				'at_ms': jnum(f64(now_ms()))
			})
			return ToolExecution{
				ok: true
				observation: bound_observation(receipt, cfg.max_observation_bytes)
				receipt: receipt
				terminal: false
				final_answer: ''
			}
		}
		'read_file' {
			rel := required_string_field(cmd.arguments, 'path')!
			full := safe_join(workspace, rel)!
			max_bytes := optional_int_field(cmd.arguments, 'max_bytes', cfg.max_tool_output_bytes)
			content := if os.exists(full) { os.read_file(full)! } else { '' }
			limited := truncate_text(content, if max_bytes > cfg.max_tool_output_bytes { cfg.max_tool_output_bytes } else { max_bytes })
			receipt := jobj({
				'type': jstr('read_file_receipt')
				'ok': jbool(true)
				'path': jstr(rel)
				'content': jstr(limited)
				'truncated': jbool(content.len > limited.len)
				'hash': jstr(sha_hex(content))
				'at_ms': jnum(f64(now_ms()))
			})
			return ToolExecution{
				ok: true
				observation: bound_observation(receipt, cfg.max_observation_bytes)
				receipt: receipt
				terminal: false
				final_answer: ''
			}
		}
		'list_files' {
			rel := optional_string_field(cmd.arguments, 'path', '')
			max_count := optional_int_field(cmd.arguments, 'max_count', 256)
			mut files := []string{}
			list_files_recursive(workspace, rel, if max_count > 1024 { 1024 } else { max_count }, mut files)!
			receipt := jobj({
				'type': jstr('list_files_receipt')
				'ok': jbool(true)
				'path': jstr(rel)
				'files': string_array_node(files)
				'at_ms': jnum(f64(now_ms()))
			})
			return ToolExecution{
				ok: true
				observation: bound_observation(receipt, cfg.max_observation_bytes)
				receipt: receipt
				terminal: false
				final_answer: ''
			}
		}
		else {
			return error('unknown tool')
		}
	}
}

fn get_path(node JsonNode, path string) ?JsonNode {
	if path.trim_space() == '' {
		return node
	}
	mut current := node
	for part in path.split('.') {
		if current.kind != .object_value {
			return none
		}
		if !(part in current.obj) {
			return none
		}
		current = current.obj[part]
	}
	return current
}

fn deterministic_output_guard(text string) VerifierResult {
	if text.contains('\x00') {
		return VerifierResult{
			ok: false
			details: 'nul byte rejected'
		}
	}
	upper := text.to_upper()
	if upper.contains('BEGIN PRIVATE KEY') || upper.contains('MODULAR_API_KEY') {
		return VerifierResult{
			ok: false
			details: 'secret-like output rejected'
		}
	}
	return VerifierResult{
		ok: true
		details: 'local guard passed'
	}
}

fn external_classify_output(cfg Config, tenant_id string, text string) VerifierResult {
	local := deterministic_output_guard(text)
	if !local.ok {
		return local
	}
	if cfg.output_classifier_url.trim_space() == '' {
		return local
	}
	payload := ClassifierRequest{
		tenant_id: tenant_id
		text: text
	}
	mut req := http.Request{
		method: .post
		url: cfg.output_classifier_url
		data: json.encode(payload)
	}
	req.add_header(.content_type, 'application/json')
	resp := req.do() or {
		return VerifierResult{
			ok: false
			details: 'classifier request failed: ${err}'
		}
	}
	if resp.status_code < 200 || resp.status_code >= 300 {
		return VerifierResult{
			ok: false
			details: 'classifier status ${resp.status_code}'
		}
	}
	parsed := json.decode(ClassifierResponse, resp.body) or {
		return VerifierResult{
			ok: false
			details: 'classifier json failed: ${err}'
		}
	}
	if !parsed.allowed {
		return VerifierResult{
			ok: false
			details: if parsed.reason == '' { 'classifier rejected output' } else { parsed.reason }
		}
	}
	return VerifierResult{
		ok: true
		details: if parsed.reason == '' { 'classifier accepted output' } else { parsed.reason }
	}
}

fn eval_verifier_node(cfg Config, tenant_id string, verifier JsonNode, state JsonNode, answer string, workspace string) VerifierResult {
	if verifier.kind == .null_value {
		return VerifierResult{
			ok: answer.trim_space() != ''
			details: 'default nonempty final answer verifier'
		}
	}
	if verifier.kind == .array_value {
		for item in verifier.arr {
			res := eval_verifier_node(cfg, tenant_id, item, state, answer, workspace)
			if !res.ok {
				return res
			}
		}
		return VerifierResult{
			ok: true
			details: 'array verifier passed'
		}
	}
	if verifier.kind != .object_value {
		return VerifierResult{
			ok: false
			details: 'verifier must be object or array'
		}
	}
	if verifier.obj.len == 0 {
		return VerifierResult{
			ok: answer.trim_space() != ''
			details: 'default nonempty final answer verifier'
		}
	}
	typ := optional_string_field(verifier, 'type', 'all')
	match typ {
		'final_answer_nonempty' {
			return VerifierResult{
				ok: answer.trim_space() != ''
				details: 'final answer nonempty'
			}
		}
		'answer_contains' {
			value := optional_string_field(verifier, 'value', '')
			return VerifierResult{
				ok: value == '' || answer.contains(value)
				details: 'answer_contains ' + value
			}
		}
		'state_has_key' {
			path := optional_string_field(verifier, 'path', '')
			_ := get_path(state, path) or {
				return VerifierResult{
					ok: false
					details: 'state path missing: ' + path
				}
			}
			return VerifierResult{
				ok: true
				details: 'state path present'
			}
		}
		'state_equals' {
			path := optional_string_field(verifier, 'path', '')
			expected := node_get(verifier, 'value') or { jnull() }
			actual := get_path(state, path) or {
				return VerifierResult{
					ok: false
					details: 'state path missing: ' + path
				}
			}
			return VerifierResult{
				ok: serialize_json_node(actual) == serialize_json_node(expected)
				details: 'state_equals ' + path
			}
		}
		'file_contains_lines' {
			path := optional_string_field(verifier, 'path', '')
			lines_node := node_get(verifier, 'lines') or { jarr([]JsonNode{}) }
			lines := node_string_array(lines_node) or { []string{} }
			full := safe_join(workspace, path) or {
				return VerifierResult{
					ok: false
					details: 'unsafe verifier file path'
				}
			}
			result := check_lines_exact(full, lines) or {
				return VerifierResult{
					ok: false
					details: 'file check failed: ${err}'
				}
			}
			for line in lines {
				if !(line in result) || !result[line] {
					return VerifierResult{
						ok: false
						details: 'missing exact line'
					}
				}
			}
			return VerifierResult{
				ok: true
				details: 'file lines present'
			}
		}
		'external_classifier' {
			return external_classify_output(cfg, tenant_id, answer)
		}
		'all' {
			checks := node_get(verifier, 'checks') or { jarr([]JsonNode{}) }
			if checks.kind != .array_value {
				return VerifierResult{
					ok: false
					details: 'all verifier requires checks array'
				}
			}
			for item in checks.arr {
				res := eval_verifier_node(cfg, tenant_id, item, state, answer, workspace)
				if !res.ok {
					return res
				}
			}
			return VerifierResult{
				ok: true
				details: 'all checks passed'
			}
		}
		'any' {
			checks := node_get(verifier, 'checks') or { jarr([]JsonNode{}) }
			if checks.kind != .array_value {
				return VerifierResult{
					ok: false
					details: 'any verifier requires checks array'
				}
			}
			mut last := 'no checks'
			for item in checks.arr {
				res := eval_verifier_node(cfg, tenant_id, item, state, answer, workspace)
				if res.ok {
					return res
				}
				last = res.details
			}
			return VerifierResult{
				ok: false
				details: last
			}
		}
		else {
			return VerifierResult{
				ok: false
				details: 'unknown verifier type: ' + typ
			}
		}
	}
}

fn verify_terminal(cfg Config, tenant_id string, session SessionRecord, answer string, state_json string) VerifierResult {
	classifier := external_classify_output(cfg, tenant_id, answer)
	if !classifier.ok {
		return classifier
	}
	verifier := parse_json_node(session.verifier_json) or { jobj(map[string]JsonNode{}) }
	state := parse_json_node(state_json) or { jobj(map[string]JsonNode{}) }
	return eval_verifier_node(cfg, tenant_id, verifier, state, answer, session.workspace_path)
}

fn sinusoidal_staleness(delta_ms i64, dims int) []f64 {
	mut result := []f64{}
	if dims <= 0 {
		return result
	}
	seconds := f64(delta_ms) / 1000.0
	for i := 0; i < dims; i++ {
		denom := math.pow(10000.0, f64(i) / f64(dims))
		if i % 2 == 0 {
			result << math.sin(seconds / denom)
		} else {
			result << math.cos(seconds / denom)
		}
	}
	return result
}

fn latest_cognition_context(mut db sqlite.DB, tenant_id string, session_id string) JsonNode {
	rows := db.exec_param_many('select cognition_json, generated_at_ms from cognition_frames where tenant_id = ? and session_id = ? order by generated_at_ms desc limit 1', [
		tenant_id,
		session_id,
	]) or { []sqlite.Row{} }
	if rows.len == 0 {
		return jobj({
			'available': jbool(false)
			'staleness_encoding': jarr([]JsonNode{})
		})
	}
	generated_at := parse_i64_default(row_get(rows[0], 1), now_ms())
	stale := now_ms() - generated_at
	mut enc := []JsonNode{}
	for value in sinusoidal_staleness(stale, 16) {
		enc << jnum(value)
	}
	return jobj({
		'available': jbool(true)
		'cognition': parse_json_node(row_get(rows[0], 0)) or { jobj(map[string]JsonNode{}) }
		'staleness_ms': jnum(f64(stale))
		'staleness_encoding': jarr(enc)
	})
}

fn environment_state_patch(cmd ActionCommand, result ToolExecution, step int) JsonNode {
	return jobj({
		'environment': jobj({
			'last_tool': jstr(cmd.tool)
			'last_tool_ok': jbool(result.ok)
			'last_step': jnum(f64(step))
			'last_observation_hash': jstr(sha_hex(serialize_json_node(result.observation)))
			'updated_at_ms': jnum(f64(now_ms()))
		})
	})
}

fn find_idempotent_result(mut db sqlite.DB, tenant_id string, session_id string, idempotency_key string, current_action_id string) ?JsonNode {
	if idempotency_key == '' {
		return none
	}
	rows := db.exec_param_many('select result_json from actions where tenant_id = ? and session_id = ? and idempotency_key = ? and id != ? and status = ? order by updated_at_ms desc limit 1', [
		tenant_id,
		session_id,
		idempotency_key,
		current_action_id,
		'done',
	]) or { return none }
	if rows.len == 0 {
		return none
	}
	return parse_json_node(row_get(rows[0], 0)) or { return none }
}

fn process_action(cfg Config, action_id string) ! {
	mut db_mark := open_db(cfg)!
	action := load_action(mut db_mark, action_id)!
	if action.status != 'pending' {
		db_mark.close() or {}
		return
	}
	db_mark.exec_param_many('update actions set status = ?, updated_at_ms = ? where id = ? and status = ?', [
		'running',
		now_ms().str(),
		action_id,
		'pending',
	])!
	session := load_session(mut db_mark, action.tenant_id, action.session_id)!
	action_node := parse_json_node(action.action_json)!
	cmd := action_command_from_node(action_node)!
	authorized := authorize_tool(cfg, mut db_mark, action.tenant_id, action.session_id, cmd)!
	cognition_context := latest_cognition_context(mut db_mark, action.tenant_id, action.session_id)
	prior_result := find_idempotent_result(mut db_mark, action.tenant_id, action.session_id, cmd.idempotency_key, action_id)
	db_mark.close() or {}
	mut tool_result := ToolExecution{}
	if prior_result != none {
		prior := prior_result?
		obs := node_get(prior, 'observation') or { jobj(map[string]JsonNode{}) }
		rcpt := node_get(prior, 'receipt') or { jobj(map[string]JsonNode{}) }
		ans := optional_string_field(prior, 'final_answer', '')
		tool_result = ToolExecution{
			ok: optional_bool_field(prior, 'ok', true)
			observation: obs
			receipt: rcpt
			terminal: ans != ''
			final_answer: ans
		}
	} else if authorized {
		tool_result = execute_tool(cfg, cmd, session.workspace_path) or {
			obs := jobj({
				'type': jstr('tool_execution_error')
				'tool': jstr(cmd.tool)
				'ok': jbool(false)
				'message': jstr('${err}')
				'at_ms': jnum(f64(now_ms()))
			})
			ToolExecution{
				ok: false
				observation: obs
				receipt: obs
				terminal: false
				final_answer: ''
			}
		}
	} else {
		tool_result = unauthorized_tool_result(cmd)
	}
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	db.exec('begin immediate transaction')!
	fresh := load_session(mut db, action.tenant_id, action.session_id)!
	env_patch := environment_state_patch(cmd, tool_result, action.step)
	new_state_json := apply_patch_json(fresh.state_json, env_patch, cfg.max_state_bytes) or { fresh.state_json }
	mut final_status := fresh.status
	mut error_message := fresh.error_message
	mut verifier_result := VerifierResult{
		ok: false
		details: 'not terminal'
	}
	if cmd.tool == 'final_answer' && tool_result.ok {
		verifier_result = verify_terminal(cfg, action.tenant_id, fresh, tool_result.final_answer, new_state_json)
		if verifier_result.ok {
			final_status = 'completed'
			error_message = ''
		} else {
			final_status = 'running'
			error_message = 'terminal verifier rejected answer: ' + verifier_result.details
			tool_result = ToolExecution{
				ok: false
				observation: jobj({
					'type': jstr('terminal_verifier_failed')
					'ok': jbool(false)
					'answer': jstr(tool_result.final_answer)
					'details': jstr(verifier_result.details)
					'at_ms': jnum(f64(now_ms()))
				})
				receipt: tool_result.receipt
				terminal: false
				final_answer: tool_result.final_answer
			}
		}
	}
	if final_status == 'running' && fresh.last_step >= fresh.max_steps {
		final_status = 'failed'
		error_message = 'maximum step count reached'
	}
	result_json := serialize_json_node(jobj({
		'ok': jbool(tool_result.ok)
		'observation': tool_result.observation
		'receipt': tool_result.receipt
		'final_answer': jstr(tool_result.final_answer)
		'verifier': jobj({
			'ok': jbool(verifier_result.ok)
			'details': jstr(verifier_result.details)
		})
		'cognition_context': cognition_context
	}))
	db.exec_param_many('update actions set status = ?, result_json = ?, error = ?, updated_at_ms = ? where id = ?', [
		'done',
		result_json,
		if tool_result.ok { '' } else { optional_string_field(tool_result.observation, 'message', '') },
		now_ms().str(),
		action_id,
	])!
	observation_json := serialize_json_node(bound_observation(tool_result.observation, cfg.max_observation_bytes))
	db.exec_param_many('update sessions set state_json = ?, observation_json = ?, status = ?, error = ?, lease_expires_ms = 0, updated_at_ms = ? where tenant_id = ? and id = ?', [
		new_state_json,
		observation_json,
		final_status,
		error_message,
		now_ms().str(),
		action.tenant_id,
		action.session_id,
	])!
	insert_observation(mut db, action.tenant_id, action.session_id, action.step, observation_json)!
	insert_checkpoint(mut db, action.tenant_id, action.session_id, action.step, 'system1', new_state_json,
		fresh.wm_json, observation_json, serialize_json_node(env_patch), action.action_json)!
	insert_raw_trace(mut db, action.tenant_id, action.session_id, jobj({
		'type': jstr('verified_system1_execution')
		'step': jnum(f64(action.step))
		'action': action_node
		'outcome_ok': jbool(tool_result.ok)
		'observation_hash': jstr(sha_hex(observation_json))
		'post_execution_state_hash': jstr(sha_hex(new_state_json))
		'receipt': tool_result.receipt
		'cognition_context': cognition_context
	}))!
	db.exec('commit')!
	if (final_status == 'completed' || final_status == 'failed') && cfg.enable_learning {
		post_trajectory_learning(cfg, action.tenant_id, action.session_id) or { eprintln('learning error: ${err}') }
	}
}

fn recover_stale_actions(mut db sqlite.DB) ! {
	cutoff := now_ms() - 60000
	db.exec_param_many('update actions set status = ?, updated_at_ms = ? where status = ? and updated_at_ms < ?', [
		'pending',
		now_ms().str(),
		'running',
		cutoff.str(),
	])!
}

fn process_system1_tick(cfg Config) ! {
	mut db := open_db(cfg)!
	recover_stale_actions(mut db)!
	rows := db.exec_param_many('select id from actions where status = ? order by created_at_ms asc limit 16', [
		'pending',
	])!
	db.close() or {}
	for row in rows {
		process_action(cfg, row_get(row, 0)) or { eprintln('system1 action error: ${err}') }
	}
}

fn system1_loop(cfg Config) {
	for {
		process_system1_tick(cfg) or { eprintln('system1 tick error: ${err}') }
		time.sleep(50 * time.millisecond)
	}
}

fn call_scored_completion(cfg Config, messages []ChatMessage, max_tokens int) !ScoredCompletion {
	payload := ChatCompletionScoringRequest{
		model: cfg.model
		messages: messages
		stream: false
		temperature: 0.0
		top_p: 1.0
		max_tokens: max_tokens
		logprobs: true
		top_logprobs: 5
		seed: 1234
	}
	mut req := http.Request{
		method: .post
		url: cfg.base_url + '/chat/completions'
		data: json.encode(payload)
	}
	req.add_header(.content_type, 'application/json')
	req.add_header(.authorization, 'Bearer ' + cfg.api_key)
	resp := req.do()!
	if resp.status_code < 200 || resp.status_code >= 300 {
		return error('scoring api error ${resp.status_code}: ' + truncate_text(resp.body, 2000))
	}
	root := parse_json_node(resp.body)!
	choices := node_get(root, 'choices') or { return error('scoring response missing choices') }
	if choices.kind != .array_value || choices.arr.len == 0 {
		return error('scoring response empty choices')
	}
	choice := choices.arr[0]
	message := node_get(choice, 'message') or { jobj(map[string]JsonNode{}) }
	content := optional_string_field(message, 'content', '')
	logprobs_node := node_get(choice, 'logprobs') or { jobj(map[string]JsonNode{}) }
	content_probs := node_get(logprobs_node, 'content') or { jarr([]JsonNode{}) }
	mut tokens := []ScoredToken{}
	if content_probs.kind == .array_value {
		for item in content_probs.arr {
			token := optional_string_field(item, 'token', '')
			lp := optional_f64_field(item, 'logprob', -100.0)
			top_node := node_get(item, 'top_logprobs') or { jarr([]JsonNode{}) }
			mut top := map[string]f64{}
			if top_node.kind == .array_value {
				for top_item in top_node.arr {
					top_token := optional_string_field(top_item, 'token', '')
					top_lp := optional_f64_field(top_item, 'logprob', -100.0)
					if top_token != '' {
						top[top_token] = top_lp
					}
				}
			}
			if token != '' && !(token in top) {
				top[token] = lp
			}
			tokens << ScoredToken{
				token: token
				logprob: lp
				top: top
			}
		}
	}
	return ScoredCompletion{
		content: content
		tokens: tokens
	}
}

fn normalized_distribution(top map[string]f64) map[string]f64 {
	mut max_lp := -1.0e300
	for _, lp in top {
		if lp > max_lp {
			max_lp = lp
		}
	}
	mut total := 0.0
	mut exp_values := map[string]f64{}
	for token, lp in top {
		value := math.exp(lp - max_lp)
		exp_values[token] = value
		total += value
	}
	mut out := map[string]f64{}
	if total <= 0.0 {
		return out
	}
	for token, value in exp_values {
		out[token] = value / total
	}
	return out
}

fn reverse_kl(student map[string]f64, teacher map[string]f64) f64 {
	ps := normalized_distribution(student)
	pt := normalized_distribution(teacher)
	mut kl := 0.0
	for token, p in ps {
		if p <= 0.0 {
			continue
		}
		q := if token in pt { pt[token] } else { 1.0e-12 }
		kl += p * (math.log(p) - math.log(q))
	}
	return kl
}

fn top_map_node(values map[string]f64) JsonNode {
	mut obj := map[string]JsonNode{}
	for key, value in values {
		obj[key] = jnum(value)
	}
	return jobj(obj)
}

fn update_policy_weight(mut db sqlite.DB, tenant_id string, key string, delta f64) ! {
	rows := db.exec_param_many('select weight from policy_weights where tenant_id = ? and key = ?', [
		tenant_id,
		key,
	])!
	mut current := 0.0
	if rows.len > 0 {
		current = strconv.atof64(row_get(rows[0], 0)) or { 0.0 }
	}
	next := current + delta
	db.exec_param_many('insert or replace into policy_weights(tenant_id, key, weight, updated_at_ms) values(?, ?, ?, ?)', [
		tenant_id,
		key,
		next.str(),
		now_ms().str(),
	])!
}

fn generate_reflection_patch(cfg Config, session SessionRecord, trace_summary string, verifier_ok bool) !string {
	system := 'You are a fixed out-of-loop reflection engine. Produce one compact JSON object diagnosing failures, pivot actions, and memory repair targets. Do not include hidden reasoning.'
	user := 'SESSION_SPEC:\n' + session.spec_json + '\nTERMINAL_STATE:\n' + session.state_json +
		'\nTERMINAL_STATUS:\n' + session.status + '\nVERIFIER_OK:\n' + verifier_ok.str() +
		'\nVERIFIED_TRACE_SUMMARY:\n' + trace_summary +
		'\nReturn JSON with fields failure_points, pivot_actions, memory_repairs, distillation_focus, success_patterns.'
	result := call_model_stream(cfg, [
		ChatMessage{
			role: 'system'
			content: system
		},
		ChatMessage{
			role: 'user'
			content: user
		},
	], 0.2, 4096, 0.0, 0.0)!
	node := parse_json_node(extract_first_json_object(result.content)!)!
	return serialize_json_node(sanitize_decision_node(node))
}

fn trajectory_trace_summary(mut db sqlite.DB, tenant_id string, session_id string) !string {
	rows := db.exec_param_many('select trace_json from raw_traces where tenant_id = ? and session_id = ? order by created_at_ms asc limit 200', [
		tenant_id,
		session_id,
	])!
	mut builder := strings.new_builder(8192)
	for row in rows {
		builder.write_string(truncate_text(row_get(row, 0), 2000))
		builder.write_u8(`\n`)
		if builder.len > 60000 {
			break
		}
	}
	return builder.str()
}

fn run_distillation_episode(cfg Config, tenant_id string, session_id string, reflection_json string) ! {
	mut db := open_db(cfg)!
	session := load_session(mut db, tenant_id, session_id)!
	rows := db.exec_param_many('select step, state_json, wm_json, observation_json from checkpoints where tenant_id = ? and session_id = ? and (phase = ? or phase = ?) order by step asc limit 64', [
		tenant_id,
		session_id,
		'initial',
		'system1',
	])!
	lessons := load_recent_playbook_lessons(cfg, tenant_id)
	policy := load_policy_signals(mut db, tenant_id)
	db.close() or {}
	for row in rows {
		step := parse_int_default(row_get(row, 0), 0)
		state_json := row_get(row, 1)
		wm_json := row_get(row, 2)
		obs_json := row_get(row, 3)
		skills := retrieve_skills(cfg, tenant_id, state_json, wm_json)!
		student_messages := build_step_messages(cfg, tenant_id, session.spec_json, state_json, wm_json, obs_json, skills, lessons, policy)
		teacher_system := student_messages[0].content + '\nPRIVILEGED_REFLECTION_PATCH:\n' + reflection_json
		teacher_messages := [
			ChatMessage{
				role: 'system'
				content: teacher_system
			},
			student_messages[1],
		]
		student := call_scored_completion(cfg, student_messages, 512) or { continue }
		teacher := call_scored_completion(cfg, teacher_messages, 512) or { continue }
		mut dbw := open_db(cfg)!
		limit := if student.tokens.len < teacher.tokens.len { student.tokens.len } else { teacher.tokens.len }
		for pos := 0; pos < limit; pos++ {
			s := student.tokens[pos]
			t := teacher.tokens[pos]
			kl := reverse_kl(s.top, t.top)
			dbw.exec_param_many('insert into distillation_tokens(id, tenant_id, session_id, step, position, student_token, teacher_token, student_logprob, teacher_logprob, reverse_kl, teacher_top_json, student_top_json, created_at_ms) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
				new_id('distill'),
				tenant_id,
				session_id,
				step.str(),
				pos.str(),
				s.token,
				t.token,
				s.logprob.str(),
				t.logprob.str(),
				kl.str(),
				serialize_json_node(top_map_node(t.top)),
				serialize_json_node(top_map_node(s.top)),
				now_ms().str(),
			])!
			update_policy_weight(mut dbw, tenant_id, 'token:' + t.token, 0.01 / (1.0 + kl))!
		}
		dbw.close() or {}
	}
}

fn post_trajectory_learning(cfg Config, tenant_id string, session_id string) ! {
	mut db := open_db(cfg)!
	existing := db.exec_param_many('select id from reflections where tenant_id = ? and session_id = ? limit 1', [
		tenant_id,
		session_id,
	])!
	if existing.len > 0 {
		db.close() or {}
		return
	}
	session := load_session(mut db, tenant_id, session_id)!
	answer := latest_final_answer(mut db, tenant_id, session_id)
	verifier := if session.status == 'completed' {
		VerifierResult{
			ok: true
			details: 'completed'
		}
	} else {
		verify_terminal(cfg, tenant_id, session, answer, session.state_json)
	}
	trace_summary := trajectory_trace_summary(mut db, tenant_id, session_id)!
	db.close() or {}
	reflection_json := generate_reflection_patch(cfg, session, trace_summary, verifier.ok) or {
		serialize_json_node(jobj({
			'failure_points': jarr([jstr('${err}')])
			'pivot_actions': jarr([]JsonNode{})
			'memory_repairs': jarr([]JsonNode{})
			'distillation_focus': jarr([]JsonNode{})
			'success_patterns': jarr([]JsonNode{})
		}))
	}
	mut db2 := open_db(cfg)!
	db2.exec_param_many('insert into reflections(id, tenant_id, session_id, reflection_json, verifier_ok, processed, created_at_ms) values(?, ?, ?, ?, ?, ?, ?)', [
		new_id('reflection'),
		tenant_id,
		session_id,
		reflection_json,
		if verifier.ok { '1' } else { '0' },
		'0',
		now_ms().str(),
	])!
	db2.close() or {}
	update_knowledge_playbook(cfg, tenant_id, reflection_json) or { eprintln('knowledge update error: ${err}') }
	run_distillation_episode(cfg, tenant_id, session_id, reflection_json) or { eprintln('distillation error: ${err}') }
}

fn append_lines_unique_path(path string, lines []string) ! {
	_ := append_lines_file(path, lines, true)!
}

fn update_knowledge_playbook(cfg Config, tenant_id string, reflection_json string) ! {
	tenant := sanitize_identifier(tenant_id)!
	root := os.join_path(cfg.knowledge_root, tenant)
	os.mkdir_all(root)!
	path := os.join_path(root, 'playbook.md')
	entry := '- ' + now_ms().str() + ' ' + truncate_text(reflection_json.replace('\n', ' '), 4000)
	append_lines_unique_path(path, [entry])!
	if !os.exists(os.join_path(root, '.git')) {
		os.execute('git -C ' + root + ' init')
	}
	diff := os.execute('git -C ' + root + ' diff -- .').output
	os.execute('git -C ' + root + ' add .')
	os.execute('git -C ' + root + ' commit -m runtime-' + now_ms().str())
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	db.exec_param_many('insert into knowledge_versions(id, tenant_id, path, diff, created_at_ms) values(?, ?, ?, ?, ?)', [
		new_id('knowledge'),
		tenant,
		path,
		diff,
		now_ms().str(),
	])!
}

fn propose_skill_patch(cfg Config, tenant_id string, reflection_json string) !JsonNode {
	system := 'You are a non-self-modifying meta-agent. Read diagnostics and propose a minimal active skill layer patch. Output one JSON object only.'
	user := 'TENANT:\n' + tenant_id + '\nREFLECTION_PATCH:\n' + reflection_json +
		'\nReturn schema {"component":"active_skill_layer","operation":"upsert_skill","skill":{"id":"","name":"","description":"","trigger":{},"body":"","tests":[]},"diagnosis":""}.'
	result := call_model_stream(cfg, [
		ChatMessage{
			role: 'system'
			content: system
		},
		ChatMessage{
			role: 'user'
			content: user
		},
	], 0.2, 4096, 0.0, 0.0)!
	node := parse_json_node(extract_first_json_object(result.content)!)!
	if node.kind != .object_value {
		return error('meta patch is not object')
	}
	return node
}

fn skill_from_patch_node(tenant_id string, skill_node JsonNode) !Skill {
	if skill_node.kind != .object_value {
		return error('skill patch skill is not object')
	}
	name := required_string_field(skill_node, 'name')!
	description := required_string_field(skill_node, 'description')!
	body := required_string_field(skill_node, 'body')!
	if name.len > 200 || description.len > 2000 || body.len > 12000 {
		return error('skill patch exceeds size limit')
	}
	if body.to_lower().contains('ignore validation') || body.to_lower().contains('bypass authorization') {
		return error('skill body violates governance terms')
	}
	mut id := optional_string_field(skill_node, 'id', '')
	if id == '' {
		id = 'skill_' + sha_hex(name + body)[..24]
	}
	_ := sanitize_identifier(id)!
	trigger := node_get(skill_node, 'trigger') or { jobj(map[string]JsonNode{}) }
	tests := node_get(skill_node, 'tests') or { jarr([]JsonNode{}) }
	return Skill{
		id: id
		tenant_id: tenant_id
		name: name
		description: description
		trigger_json: serialize_json_node(trigger)
		body: body
		embedding_json: '[]'
		version: 1
		enabled: 1
		tests_json: serialize_json_node(tests)
		updated_at_ms: now_ms()
	}
}

fn run_skill_diagnostic(cfg Config, tenant_id string, task string, verifier_json string, skills []Skill) !bool {
	req := ApiChatRequest{
		task: task
	}
	spec_json := initial_spec_json(task, req, verifier_json)
	state_json := initial_state_json(task)
	wm_json := initial_wm_json(task, []string{})
	obs_json := initial_observation_json(task)
	temp_workspace := os.join_path(cfg.workspace_root, 'diagnostic_' + new_id('workspace'))
	os.mkdir_all(temp_workspace)!
	defer {
		os.rmdir_all(temp_workspace) or {}
	}
	mut session := SessionRecord{
		id: new_id('diagnostic_session')
		tenant_id: tenant_id
		spec_json: spec_json
		state_json: state_json
		wm_json: wm_json
		status: 'running'
		created_at_ms: now_ms()
		updated_at_ms: now_ms()
		lease_expires_ms: 0
		last_step: 0
		max_steps: 4
		token_budget: cfg.default_token_budget
		tokens_used: 0
		observation_json: obs_json
		workspace_path: temp_workspace
		verifier_json: verifier_json
		error_message: ''
	}
	lessons := load_recent_playbook_lessons(cfg, tenant_id)
	for session.last_step < session.max_steps && session.status == 'running' {
		messages := build_step_messages(cfg, tenant_id, session.spec_json, session.state_json, session.wm_json, session.observation_json, skills, lessons, '[]')
		result := call_model_stream(cfg, messages, 0.1, 4096, 0.0, 0.0) or { return false }
		decision := decode_step_decision(result.content) or { return false }
		transition := validate_transition(cfg, session, decision) or { return false }
		cmd := decision.action
		if decision.terminal || cmd.tool == 'final_answer' {
			answer := optional_string_field(cmd.arguments, 'answer', '')
			state := parse_json_node(transition.new_state_json) or { return false }
			verifier := parse_json_node(verifier_json) or { jobj(map[string]JsonNode{}) }
			res := eval_verifier_node(cfg, tenant_id, verifier, state, answer, temp_workspace)
			return res.ok
		}
		if cmd.tool != 'noop' {
			tool_result := execute_tool(cfg, cmd, temp_workspace) or { return false }
			env_patch := environment_state_patch(cmd, tool_result, session.last_step + 1)
			session.state_json = apply_patch_json(transition.new_state_json, env_patch, cfg.max_state_bytes) or { return false }
			session.observation_json = serialize_json_node(tool_result.observation)
		} else {
			session.state_json = transition.new_state_json
		}
		session.wm_json = transition.new_wm_json
		session.last_step++
	}
	return false
}

fn validate_skill_patch(cfg Config, tenant_id string, proposal JsonNode) !SkillPatchValidation {
	component := optional_string_field(proposal, 'component', '')
	if component != 'active_skill_layer' {
		return SkillPatchValidation{
			ok: false
			details: 'component is not active_skill_layer'
		}
	}
	operation := optional_string_field(proposal, 'operation', '')
	if operation != 'upsert_skill' && operation != 'disable_skill' {
		return SkillPatchValidation{
			ok: false
			details: 'unsupported skill operation'
		}
	}
	if operation == 'disable_skill' {
		return SkillPatchValidation{
			ok: true
			details: 'disable operation schema accepted'
		}
	}
	skill_node := node_get(proposal, 'skill') or {
		return SkillPatchValidation{
			ok: false
			details: 'missing skill'
		}
	}
	candidate := skill_from_patch_node(tenant_id, skill_node)!
	mut candidate_skills := [candidate]
	tests_node := parse_json_node(candidate.tests_json) or { jarr([]JsonNode{}) }
	if tests_node.kind == .array_value {
		for test in tests_node.arr {
			task := optional_string_field(test, 'task', '')
			verifier_node := node_get(test, 'verifier') or { jobj(map[string]JsonNode{}) }
			if task.trim_space() == '' {
				continue
			}
			ok := run_skill_diagnostic(cfg, tenant_id, task, serialize_json_node(verifier_node), candidate_skills) or {
				return SkillPatchValidation{
					ok: false
					details: 'candidate diagnostic failed: ${err}'
				}
			}
			if !ok {
				return SkillPatchValidation{
					ok: false
					details: 'candidate test verifier failed'
				}
			}
		}
	}
	mut db := open_db(cfg)!
	rows := db.exec_param_many('select task, verifier_json from diagnostic_tasks where tenant_id = ? or tenant_id = ? order by created_at_ms asc limit ?', [
		tenant_id,
		'global',
		cfg.max_diagnostic_tasks.str(),
	])!
	db.close() or {}
	for row in rows {
		task := row_get(row, 0)
		verifier_json := row_get(row, 1)
		baseline_skills := retrieve_skills(cfg, tenant_id, initial_state_json(task), initial_wm_json(task,
			[]string{})) or { []Skill{} }
		baseline_ok := run_skill_diagnostic(cfg, tenant_id, task, verifier_json, baseline_skills) or { false }
		mut merged := baseline_skills.clone()
		merged << candidate
		candidate_ok := run_skill_diagnostic(cfg, tenant_id, task, verifier_json, merged) or { false }
		if baseline_ok && !candidate_ok {
			return SkillPatchValidation{
				ok: false
				details: 'held-out regression detected'
			}
		}
	}
	return SkillPatchValidation{
		ok: true
		details: 'validation passed'
	}
}

fn apply_skill_patch(cfg Config, tenant_id string, proposal JsonNode) ! {
	operation := optional_string_field(proposal, 'operation', '')
	skill_node := node_get(proposal, 'skill') or { jobj(map[string]JsonNode{}) }
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	db.exec('begin immediate transaction')!
	if operation == 'disable_skill' {
		id := optional_string_field(skill_node, 'id', '')
		_ := sanitize_identifier(id)!
		db.exec_param_many('update skills set enabled = ?, updated_at_ms = ? where tenant_id = ? and id = ?', [
			'0',
			now_ms().str(),
			tenant_id,
			id,
		])!
		refresh_skill_fts(mut db, tenant_id, id)!
	} else if operation == 'upsert_skill' {
		mut skill := skill_from_patch_node(tenant_id, skill_node)!
		embedding := call_embedding(cfg, skill.name + '\n' + skill.description + '\n' + skill.body) or { []f64{} }
		skill.embedding_json = encode_vector(embedding)
		existing := db.exec_param_many('select version from skills where tenant_id = ? and id = ?', [
			tenant_id,
			skill.id,
		])!
		version := if existing.len > 0 { parse_int_default(row_get(existing[0], 0), 1) + 1 } else { 1 }
		db.exec_param_many('insert or replace into skills(id, tenant_id, name, description, trigger_json, body, embedding_json, version, enabled, tests_json, created_at_ms, updated_at_ms) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
			skill.id,
			tenant_id,
			skill.name,
			skill.description,
			skill.trigger_json,
			skill.body,
			skill.embedding_json,
			version.str(),
			'1',
			skill.tests_json,
			now_ms().str(),
			now_ms().str(),
		])!
		refresh_skill_fts(mut db, tenant_id, skill.id)!
		db.exec_param_many('insert into skill_events(id, tenant_id, skill_id, event_json, created_at_ms) values(?, ?, ?, ?, ?)', [
			new_id('skill_event'),
			tenant_id,
			skill.id,
			serialize_json_node(proposal),
			now_ms().str(),
		])!
	}
	db.exec('commit')!
}

fn run_meta_cycle(cfg Config) ! {
	mut db := open_db(cfg)!
	rows := db.exec_param_many('select id, tenant_id, reflection_json from reflections where processed = ? order by created_at_ms asc limit 4', [
		'0',
	])!
	db.close() or {}
	for row in rows {
		reflection_id := row_get(row, 0)
		tenant_id := row_get(row, 1)
		reflection_json := row_get(row, 2)
		proposal := propose_skill_patch(cfg, tenant_id, reflection_json) or {
			eprintln('meta proposal failed: ${err}')
			continue
		}
		validation := validate_skill_patch(cfg, tenant_id, proposal) or {
			eprintln('meta validation errored: ${err}')
			continue
		}
		if validation.ok {
			apply_skill_patch(cfg, tenant_id, proposal) or { eprintln('skill patch apply failed: ${err}') }
		}
		mut db2 := open_db(cfg)!
		db2.exec_param_many('update reflections set processed = ? where id = ?', ['1', reflection_id])!
		db2.close() or {}
	}
}

fn meta_agent_loop(cfg Config) {
	for {
		run_meta_cycle(cfg) or { eprintln('meta cycle error: ${err}') }
		time.sleep(cfg.meta_interval_seconds * time.second)
	}
}

fn session_events_json(cfg Config, tenant_id string, session_id string) !JsonNode {
	mut db := open_db(cfg)!
	defer {
		db.close() or {}
	}
	session := load_session(mut db, tenant_id, session_id)!
	obs_rows := db.exec_param_many('select step, obs_json, created_at_ms from observations where tenant_id = ? and session_id = ? order by created_at_ms asc limit 500', [
		tenant_id,
		session_id,
	])!
	action_rows := db.exec_param_many('select step, action_json, status, result_json, updated_at_ms from actions where tenant_id = ? and session_id = ? order by created_at_ms asc limit 500', [
		tenant_id,
		session_id,
	])!
	mut observations := []JsonNode{}
	for row in obs_rows {
		observations << jobj({
			'step': jnum(f64(parse_int_default(row_get(row, 0), 0)))
			'observation': parse_json_node(row_get(row, 1)) or { jobj(map[string]JsonNode{}) }
			'created_at_ms': jnum(f64(parse_i64_default(row_get(row, 2), 0)))
		})
	}
	mut actions := []JsonNode{}
	for row in action_rows {
		actions << jobj({
			'step': jnum(f64(parse_int_default(row_get(row, 0), 0)))
			'action': parse_json_node(row_get(row, 1)) or { jobj(map[string]JsonNode{}) }
			'status': jstr(row_get(row, 2))
			'result': parse_json_node(row_get(row, 3)) or { jobj(map[string]JsonNode{}) }
			'updated_at_ms': jnum(f64(parse_i64_default(row_get(row, 4), 0)))
		})
	}
	return jobj({
		'session_id': jstr(session_id)
		'status': jstr(session.status)
		'last_step': jnum(f64(session.last_step))
		'observations': jarr(observations)
		'actions': jarr(actions)
	})
}

fn (mut app App) before_request() {
	app.add_header('Access-Control-Allow-Origin', app.cfg.public_origin)
	app.add_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Tenant-ID')
	app.add_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
	app.add_header('Cache-Control', 'no-store')
}

fn (mut app App) current_tenant() string {
	raw := app.get_header(app.cfg.tenant_header)
	if raw.trim_space() == '' {
		return 'default'
	}
	return sanitize_identifier(raw) or { 'default' }
}

fn (mut app App) respond_error(status int, message string) vweb.Result {
	app.set_status(status, message)
	return app.json(ErrorResponse{
		error: message
	})
}

@['/'; get]
pub fn (mut app App) index() vweb.Result {
	if os.exists('index.html') {
		return app.file('index.html')
	}
	return app.text('autonomous agent runtime backend')
}

@['/healthz'; get]
pub fn (mut app App) healthz() vweb.Result {
	return app.json(jobj({
		'ok': jbool(true)
		'model': jstr(app.cfg.model)
		'time_ms': jnum(f64(now_ms()))
	}))
}

@['/api/chat'; options]
pub fn (mut app App) api_chat_options() vweb.Result {
	return app.text('')
}

@['/api/chat'; post]
pub fn (mut app App) api_chat() vweb.Result {
	body := app.req.data
	req := json.decode(ApiChatRequest, body) or {
		ApiChatRequest{
			message: body
		}
	}
	task := extract_task_from_request(req)
	if task == '' && req.session_id == '' {
		return app.respond_error(400, 'message or task is required')
	}
	tenant_id := app.current_tenant()
	if req.session_id.trim_space() != '' {
		continue_session(app.cfg, tenant_id, req.session_id.trim_space(), task) or {
			return app.respond_error(500, '${err}')
		}
		resp := wait_for_session(app.cfg, tenant_id, req.session_id.trim_space(), req.wait_ms) or {
			return app.respond_error(500, '${err}')
		}
		return app.json(resp)
	}
	verifier_json := verifier_from_body(body, req) or {
		return app.respond_error(400, 'invalid verifier json: ${err}')
	}
	session_id := create_session(app.cfg, tenant_id, req, task, verifier_json) or {
		return app.respond_error(500, '${err}')
	}
	resp := wait_for_session(app.cfg, tenant_id, session_id, req.wait_ms) or {
		return app.respond_error(500, '${err}')
	}
	return app.json(resp)
}

@['/v1/chat/completions'; options]
pub fn (mut app App) v1_options() vweb.Result {
	return app.text('')
}

@['/v1/chat/completions'; post]
pub fn (mut app App) v1_chat_completions() vweb.Result {
	body := app.req.data
	open_req := json.decode(OpenAIChatRequest, body) or {
		return app.respond_error(400, 'invalid chat completions request')
	}
	api_req := ApiChatRequest{
		messages: open_req.messages
		wait_ms: 0
	}
	task := extract_task_from_request(api_req)
	if task == '' {
		return app.respond_error(400, 'user message is required')
	}
	tenant_id := app.current_tenant()
	verifier_json := '{}'
	session_id := create_session(app.cfg, tenant_id, api_req, task, verifier_json) or {
		return app.respond_error(500, '${err}')
	}
	content := 'Autonomous session ' + session_id + ' started. Poll /api/sessions/' + session_id + '/state and /api/sessions/' + session_id + '/events.'
	resp := OpenAICompatibleResponse{
		id: 'chatcmpl_' + session_id
		object: 'chat.completion'
		created: int(now_seconds())
		model: app.cfg.model
		choices: [
			OpenAIChoice{
				index: 0
				message: ChatMessage{
					role: 'assistant'
					content: content
				}
				finish_reason: 'stop'
			},
		]
		usage: Usage{
			prompt_tokens: estimate_tokens(body)
			completion_tokens: estimate_tokens(content)
			total_tokens: estimate_tokens(body) + estimate_tokens(content)
		}
	}
	return app.json(resp)
}

@['/api/sessions/:id/state'; get]
pub fn (mut app App) api_session_state(id string) vweb.Result {
	tenant_id := app.current_tenant()
	mut db := open_db(app.cfg) or { return app.respond_error(500, '${err}') }
	defer {
		db.close() or {}
	}
	session := load_session(mut db, tenant_id, id) or { return app.respond_error(404, '${err}') }
	answer := latest_final_answer(mut db, tenant_id, id)
	return app.json(jobj({
		'session_id': jstr(id)
		'tenant_id': jstr(tenant_id)
		'status': jstr(session.status)
		'last_step': jnum(f64(session.last_step))
		'max_steps': jnum(f64(session.max_steps))
		'tokens_used': jnum(f64(session.tokens_used))
		'token_budget': jnum(f64(session.token_budget))
		'specification': parse_json_node(session.spec_json) or { jobj(map[string]JsonNode{}) }
		'state': parse_json_node(session.state_json) or { jobj(map[string]JsonNode{}) }
		'working_memory': parse_json_node(session.wm_json) or { jobj(map[string]JsonNode{}) }
		'latest_observation': parse_json_node(session.observation_json) or { jobj(map[string]JsonNode{}) }
		'answer': jstr(answer)
		'error': jstr(session.error_message)
	}))
}

@['/api/session/:id/state'; get]
pub fn (mut app App) api_session_state_alias(id string) vweb.Result {
	return app.api_session_state(id)
}

@['/api/sessions/:id/events'; get]
pub fn (mut app App) api_session_events(id string) vweb.Result {
	tenant_id := app.current_tenant()
	node := session_events_json(app.cfg, tenant_id, id) or { return app.respond_error(404, '${err}') }
	return app.json(node)
}

@['/api/sessions/:id/observe'; post]
pub fn (mut app App) api_session_observe(id string) vweb.Result {
	tenant_id := app.current_tenant()
	req := json.decode(ApiChatRequest, app.req.data) or {
		ApiChatRequest{
			message: app.req.data
		}
	}
	task := extract_task_from_request(req)
	if task == '' {
		return app.respond_error(400, 'observation content is required')
	}
	continue_session(app.cfg, tenant_id, id, task) or { return app.respond_error(500, '${err}') }
	resp := response_for_session(app.cfg, tenant_id, id) or { return app.respond_error(500, '${err}') }
	return app.json(resp)
}

@['/api/sessions/:id/stop'; post]
pub fn (mut app App) api_session_stop(id string) vweb.Result {
	tenant_id := app.current_tenant()
	mut db := open_db(app.cfg) or { return app.respond_error(500, '${err}') }
	defer {
		db.close() or {}
	}
	db.exec_param_many('update sessions set status = ?, lease_expires_ms = 0, updated_at_ms = ? where tenant_id = ? and id = ?', [
		'paused',
		now_ms().str(),
		tenant_id,
		id,
	]) or { return app.respond_error(500, '${err}') }
	return app.json(CreateSessionResponse{
		session_id: id
		status: 'paused'
		answer: latest_final_answer(mut db, tenant_id, id)
		state_url: '/api/sessions/${id}/state'
		events_url: '/api/sessions/${id}/events'
	})
}

@['/api/sessions/:id/resume'; post]
pub fn (mut app App) api_session_resume(id string) vweb.Result {
	tenant_id := app.current_tenant()
	mut db := open_db(app.cfg) or { return app.respond_error(500, '${err}') }
	defer {
		db.close() or {}
	}
	db.exec_param_many('update sessions set status = ?, error = ?, lease_expires_ms = 0, updated_at_ms = ? where tenant_id = ? and id = ?', [
		'running',
		'',
		now_ms().str(),
		tenant_id,
		id,
	]) or { return app.respond_error(500, '${err}') }
	return app.json(CreateSessionResponse{
		session_id: id
		status: 'running'
		answer: latest_final_answer(mut db, tenant_id, id)
		state_url: '/api/sessions/${id}/state'
		events_url: '/api/sessions/${id}/events'
	})
}

@['/api/sessions/:id/authorize_tool'; post]
pub fn (mut app App) api_authorize_tool(id string) vweb.Result {
	tenant_id := app.current_tenant()
	req := json.decode(ToolAuthRequest, app.req.data) or { return app.respond_error(400, 'invalid authorization request') }
	if req.tool_name == '' {
		return app.respond_error(400, 'tool_name is required')
	}
	expires := now_ms() + i64(if req.expires_in_seconds > 0 { req.expires_in_seconds } else { 3600 }) * 1000
	mut db := open_db(app.cfg) or { return app.respond_error(500, '${err}') }
	defer {
		db.close() or {}
	}
	_ := load_session(mut db, tenant_id, id) or { return app.respond_error(404, '${err}') }
	db.exec_param_many('insert into tool_authorizations(id, tenant_id, session_id, tool_name, path_prefix, expires_at_ms, created_at_ms) values(?, ?, ?, ?, ?, ?, ?)', [
		new_id('auth'),
		tenant_id,
		id,
		req.tool_name,
		req.path_prefix,
		expires.str(),
		now_ms().str(),
	]) or { return app.respond_error(500, '${err}') }
	return app.json(jobj({
		'ok': jbool(true)
		'expires_at_ms': jnum(f64(expires))
	}))
}

@['/api/skills'; get]
pub fn (mut app App) api_skills() vweb.Result {
	tenant_id := app.current_tenant()
	mut db := open_db(app.cfg) or { return app.respond_error(500, '${err}') }
	defer {
		db.close() or {}
	}
	rows := db.exec_param_many('select id, tenant_id, name, description, trigger_json, body, embedding_json, version, enabled, tests_json, updated_at_ms from skills where tenant_id = ? or tenant_id = ? order by updated_at_ms desc limit 200', [
		tenant_id,
		'global',
	]) or { return app.respond_error(500, '${err}') }
	mut arr := []JsonNode{}
	for row in rows {
		skill := skill_from_row(row)
		arr << jobj({
			'id': jstr(skill.id)
			'tenant_id': jstr(skill.tenant_id)
			'name': jstr(skill.name)
			'description': jstr(skill.description)
			'trigger': parse_json_node(skill.trigger_json) or { jobj(map[string]JsonNode{}) }
			'version': jnum(f64(skill.version))
			'enabled': jbool(skill.enabled == 1)
			'tests': parse_json_node(skill.tests_json) or { jarr([]JsonNode{}) }
			'updated_at_ms': jnum(f64(skill.updated_at_ms))
		})
	}
	return app.json(jobj({
		'skills': jarr(arr)
	}))
}

@['/api/skills'; post]
pub fn (mut app App) api_skills_post() vweb.Result {
	tenant_id := app.current_tenant()
	proposal := parse_json_node(app.req.data) or { return app.respond_error(400, 'invalid json') }
	validation := validate_skill_patch(app.cfg, tenant_id, proposal) or {
		return app.respond_error(400, '${err}')
	}
	if !validation.ok {
		return app.respond_error(400, validation.details)
	}
	apply_skill_patch(app.cfg, tenant_id, proposal) or { return app.respond_error(500, '${err}') }
	return app.json(jobj({
		'ok': jbool(true)
		'details': jstr(validation.details)
	}))
}

fn main() {
	cfg := load_config()
	init_db(cfg) or { panic(err) }
	spawn system2_loop(cfg)
	spawn system1_loop(cfg)
	if cfg.enable_meta_agent {
		spawn meta_agent_loop(cfg)
	}
	mut app := &App{
		cfg: cfg
	}
	vweb.run(app, cfg.port)
}
