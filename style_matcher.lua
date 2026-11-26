--[[
    Style Matcher — Aegisub Automation 4 Lua Macro
    Extracts exact primary fill color (\1c), border color (\3c), edge blur (\blur),
    and film grain opacity/vector mask from a video frame using a 2-point \clip probe.

    Workflow:
      1. Create or select a subtitle line and navigate to the clearest frame of the sign.
      2. Draw a 2-point clip over the sign:
         - Vector clip: \clip(m x1 y1 l x2 y2)
         - Or rectangular clip: \clip(x1,y1,x2,y2)
         Point 1 (x1, y1) inside the core text fill; Point 2 (x2, y2) across the
         text border out into the immediate background plate.
      3. Run Automation -> Style Matcher.
]]

script_name = "Style Matcher"
script_description = "Extracts fill color, border color, edge blur, and film grain from a video sign via a 2-point clip probe"
script_author = "Style Matcher"
script_version = "1.0.0"

-- ---------------------------------------------------------------------------
-- Lightweight JSON parser & encoder (fallback when Aegisub's json is absent)
-- ---------------------------------------------------------------------------
local json_decode, json_encode
do
    local ok, mod = pcall(require, "json")
    if ok and mod and type(mod.decode) == "function" then
        json_decode = mod.decode
    end
    if ok and mod and type(mod.encode) == "function" then
        json_encode = mod.encode
    end

    if not json_decode then
        local function skip_ws(s, i)
            local _, j = s:find("^[ \t\r\n]*", i)
            return j + 1
        end

        local parse_value

        local function parse_string(s, i)
            i = i + 1 -- skip opening quote
            local out = {}
            while i <= #s do
                local c = s:sub(i, i)
                if c == '"' then
                    return table.concat(out), i + 1
                elseif c == "\\" then
                    local nxt = s:sub(i + 1, i + 1)
                    if nxt == '"' or nxt == "\\" or nxt == "/" then
                        table.insert(out, nxt)
                        i = i + 2
                    elseif nxt == "n" then
                        table.insert(out, "\n")
                        i = i + 2
                    elseif nxt == "r" then
                        table.insert(out, "\r")
                        i = i + 2
                    elseif nxt == "t" then
                        table.insert(out, "\t")
                        i = i + 2
                    elseif nxt == "b" then
                        table.insert(out, "\b")
                        i = i + 2
                    elseif nxt == "f" then
                        table.insert(out, "\f")
                        i = i + 2
                    elseif nxt == "u" then
                        local hex = s:sub(i + 2, i + 5)
                        local cp = tonumber(hex, 16) or 32
                        if cp < 128 then
                            table.insert(out, string.char(cp))
                        else
                            table.insert(out, "?")
                        end
                        i = i + 6
                    else
                        table.insert(out, nxt)
                        i = i + 2
                    end
                else
                    table.insert(out, c)
                    i = i + 1
                end
            end
            error("Unterminated JSON string")
        end

        local function parse_number(s, i)
            local _, j, num_str = s:find("^(-?%d+%.?%d*[eE]?[+-]?%d*)", i)
            if not num_str then
                error("Invalid JSON number at position " .. i)
            end
            return tonumber(num_str), j + 1
        end

        local function parse_array(s, i)
            i = skip_ws(s, i + 1)
            local arr = {}
            if s:sub(i, i) == "]" then
                return arr, i + 1
            end
            while true do
                local val
                val, i = parse_value(s, i)
                table.insert(arr, val)
                i = skip_ws(s, i)
                local c = s:sub(i, i)
                if c == "]" then
                    return arr, i + 1
                elseif c == "," then
                    i = skip_ws(s, i + 1)
                else
                    error("Expected ',' or ']' in JSON array at " .. i)
                end
            end
        end

        local function parse_object(s, i)
            i = skip_ws(s, i + 1)
            local obj = {}
            if s:sub(i, i) == "}" then
                return obj, i + 1
            end
            while true do
                if s:sub(i, i) ~= '"' then
                    error("Expected string key in JSON object at " .. i)
                end
                local key
                key, i = parse_string(s, i)
                i = skip_ws(s, i)
                if s:sub(i, i) ~= ":" then
                    error("Expected ':' after JSON key at " .. i)
                end
                i = skip_ws(s, i + 1)
                local val
                val, i = parse_value(s, i)
                obj[key] = val
                i = skip_ws(s, i)
                local c = s:sub(i, i)
                if c == "}" then
                    return obj, i + 1
                elseif c == "," then
                    i = skip_ws(s, i + 1)
                else
                    error("Expected ',' or '}' in JSON object at " .. i)
                end
            end
        end

        parse_value = function(s, i)
            i = skip_ws(s, i)
            local c = s:sub(i, i)
            if c == '"' then
                return parse_string(s, i)
            elseif c == "{" then
                return parse_object(s, i)
            elseif c == "[" then
                return parse_array(s, i)
            elseif s:sub(i, i + 3) == "true" then
                return true, i + 4
            elseif s:sub(i, i + 4) == "false" then
                return false, i + 5
            elseif s:sub(i, i + 3) == "null" then
                return nil, i + 4
            else
                return parse_number(s, i)
            end
        end

        json_decode = function(str)
            local val = parse_value(str, 1)
            return val
        end
    end

    if not json_encode then
        local function escape_str(s)
            local replacements = {
                ['\\'] = '\\\\',
                ['"']  = '\\"',
                ['\b'] = '\\b',
                ['\f'] = '\\f',
                ['\n'] = '\\n',
                ['\r'] = '\\r',
                ['\t'] = '\\t',
            }
            return '"' .. s:gsub('[\\"\b\f\n\r\t%c]', function(c)
                return replacements[c] or string.format("\\u%04x", string.byte(c))
            end) .. '"'
        end

        local function encode_val(v, indent)
            local t = type(v)
            if v == nil then
                return "null"
            elseif t == "boolean" then
                return v and "true" or "false"
            elseif t == "number" then
                if v ~= v or v == math.huge or v == -math.huge then
                    return "null"
                end
                return tostring(v)
            elseif t == "string" then
                return escape_str(v)
            elseif t == "table" then
                local keys = {}
                local is_array = true
                local max_idx = 0
                for k in pairs(v) do
                    if type(k) == "number" and k >= 1 and math.floor(k) == k then
                        if k > max_idx then max_idx = k end
                    else
                        is_array = false
                    end
                    table.insert(keys, k)
                end
                if is_array and max_idx == #keys then
                    local parts = {}
                    for i = 1, max_idx do
                        table.insert(parts, encode_val(v[i], indent))
                    end
                    return "[" .. table.concat(parts, ", ") .. "]"
                else
                    table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
                    local pad = string.rep("  ", indent + 1)
                    local close_pad = string.rep("  ", indent)
                    local parts = {}
                    for _, k in ipairs(keys) do
                        table.insert(parts, pad .. escape_str(tostring(k)) .. ": " .. encode_val(v[k], indent + 1))
                    end
                    if #parts == 0 then
                        return "{}"
                    end
                    return "{\n" .. table.concat(parts, ",\n") .. "\n" .. close_pad .. "}"
                end
            else
                error("Cannot JSON-encode type: " .. t)
            end
        end

        json_encode = function(val)
            return encode_val(val, 0)
        end
    end
end

-- ---------------------------------------------------------------------------
-- Helper Utilities
-- ---------------------------------------------------------------------------
local function get_script_dir()
    local info = debug.getinfo(1, "S")
    local src = info and info.source or ""
    if src:sub(1, 1) == "@" then
        src = src:sub(2)
    end
    local dir = src:match("^(.*)[/\\][^/\\]+$")
    return dir or "."
end

local function is_windows()
    return package.config:sub(1, 1) == "\\"
end

local function quote_arg(val)
    local s = tostring(val)
    -- Double-quote escaping safe for Windows cmd & POSIX sh
    s = s:gsub('"', '\\"')
    return '"' .. s .. '"'
end

local function trim(s)
    if type(s) ~= "string" then
        return ""
    end
    return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- ---------------------------------------------------------------------------
-- Persistent Config (?user/config/style_matcher.json)
-- ---------------------------------------------------------------------------
local function get_config_dir()
    if aegisub and type(aegisub.decode_path) == "function" then
        local ok, resolved = pcall(aegisub.decode_path, "?user/config")
        if ok and type(resolved) == "string" and resolved ~= "" and resolved:sub(1, 1) ~= "?" then
            return (resolved:gsub("[/\\]+$", ""))
        end
    end
    if is_windows() then
        local appdata = os.getenv("APPDATA") or os.getenv("USERPROFILE") or "."
        return appdata:gsub("[/\\]+$", "") .. "\\Aegisub\\config"
    else
        local xdg = os.getenv("XDG_CONFIG_HOME")
        if xdg and xdg ~= "" then
            return xdg:gsub("[/\\]+$", "") .. "/aegisub/config"
        end
        local home = os.getenv("HOME") or "."
        return home:gsub("[/\\]+$", "") .. "/.aegisub/config"
    end
end

local function get_config_file_path()
    local sep = is_windows() and "\\" or "/"
    return get_config_dir() .. sep .. "style_matcher.json"
end

local function ensure_dir_exists(dir_path)
    if not dir_path or dir_path == "" then
        return
    end
    local ok_lfs, lfs = pcall(require, "lfs")
    if ok_lfs and lfs and type(lfs.mkdir) == "function" then
        local sep = is_windows() and "\\" or "/"
        local acc = ""
        if dir_path:sub(1, 1) == "/" then
            acc = "/"
        elseif dir_path:match("^%a:[/\\]") then
            acc = dir_path:sub(1, 3)
            dir_path = dir_path:sub(4)
        end
        for part in dir_path:gmatch("[^/\\]+") do
            if acc == "" or acc:sub(-1) == "/" or acc:sub(-1) == "\\" then
                acc = acc .. part
            else
                acc = acc .. sep .. part
            end
            lfs.mkdir(acc)
        end
        return
    end

    if is_windows() then
        os.execute('mkdir "' .. dir_path:gsub('"', '') .. '" >nul 2>&1')
    else
        os.execute("mkdir -p '" .. dir_path:gsub("'", "'\\''") .. "' >/dev/null 2>&1")
    end
end

local function default_settings()
    local sep = is_windows() and "\\" or "/"
    return {
        py_path = get_script_dir() .. sep .. "match_worker.py",
        match_primary = true,
        match_border = true,
        match_blur = true,
        match_grain = true,
        radius = 2,
    }
end

local function sanitize_settings(raw)
    local s = default_settings()
    if type(raw) ~= "table" then
        return s
    end
    if type(raw.py_path) == "string" then
        local cleaned_path = trim(raw.py_path):gsub('^"(.*)"$', "%1")
        if cleaned_path ~= "" then
            s.py_path = cleaned_path
        end
    end
    if type(raw.match_primary) == "boolean" then s.match_primary = raw.match_primary end
    if type(raw.match_border) == "boolean"  then s.match_border = raw.match_border end
    if type(raw.match_blur) == "boolean"    then s.match_blur = raw.match_blur end
    if type(raw.match_grain) == "boolean"   then s.match_grain = raw.match_grain end
    if type(raw.radius) == "number" then
        local r = math.floor(raw.radius + 0.5)
        if r < 1 then r = 1 elseif r > 5 then r = 5 end
        s.radius = r
    end
    return s
end

local function load_settings()
    local path = get_config_file_path()
    local f = io.open(path, "r")
    if not f then
        return default_settings()
    end
    local content = f:read("*a") or ""
    f:close()
    local ok, parsed = pcall(json_decode, content)
    if not ok or type(parsed) ~= "table" then
        return default_settings()
    end
    return sanitize_settings(parsed)
end

local function save_settings(settings)
    local clean = sanitize_settings(settings)
    local path = get_config_file_path()
    local f = io.open(path, "w")
    if not f then
        ensure_dir_exists(get_config_dir())
        f = io.open(path, "w")
    end
    if not f then
        return false
    end
    local ok, encoded = pcall(json_encode, {
        py_path = clean.py_path,
        match_primary = clean.match_primary,
        match_border = clean.match_border,
        match_blur = clean.match_blur,
        match_grain = clean.match_grain,
        radius = clean.radius,
    })
    if ok and type(encoded) == "string" then
        f:write(encoded .. "\n")
    end
    f:close()
    return ok
end

local function get_play_res(subtitles)
    local res_x, res_y = 0, 0
    for i = 1, #subtitles do
        local item = subtitles[i]
        if item.class == "info" then
            if item.key == "PlayResX" then
                res_x = tonumber(item.value) or 0
            elseif item.key == "PlayResY" then
                res_y = tonumber(item.value) or 0
            end
        elseif item.class == "dialogue" then
            break
        end
    end
    return res_x, res_y
end

--- Extract Point 1 (x1, y1) and Point 2 (x2, y2) from a \clip or \iclip tag.
-- Supports both vector clips \clip(m x1 y1 l x2 y2 ...) and rectangular clips \clip(x1,y1,x2,y2).
local function extract_probe_points(text)
    if not text or text == "" then
        return nil
    end

    -- 1. Vector 2-point line or multi-point clip: \clip(m x1 y1 l x2 y2 ...)
    for block in text:gmatch("{([^}]*)}") do
        local x1, y1, x2, y2 = block:match(
            "\\i?clip%(%s*[0-9]*%s*,?%s*m%s+(-?%d+%.?%d*)%s+(-?%d+%.?%d*)%s+l%s+(-?%d+%.?%d*)%s+(-?%d+%.?%d*)[^%)]*%)"
        )
        if x1 and y1 and x2 and y2 then
            return tonumber(x1), tonumber(y1), tonumber(x2), tonumber(y2)
        end

        -- Also handle optional scale parameter variant without comma before m: \clip(1, m x1 y1 l x2 y2)
        x1, y1, x2, y2 = block:match(
            "\\i?clip%(%s*m%s+(-?%d+%.?%d*)%s+(-?%d+%.?%d*)%s+l%s+(-?%d+%.?%d*)%s+(-?%d+%.?%d*)[^%)]*%)"
        )
        if x1 and y1 and x2 and y2 then
            return tonumber(x1), tonumber(y1), tonumber(x2), tonumber(y2)
        end

        -- 2. Rectangular 2-point clip: \clip(x1, y1, x2, y2)
        x1, y1, x2, y2 = block:match(
            "\\i?clip%(%s*(-?%d+%.?%d*)%s*,%s*(-?%d+%.?%d*)%s*,%s*(-?%d+%.?%d*)%s*,%s*(-?%d+%.?%d*)%s*%)"
        )
        if x1 and y1 and x2 and y2 then
            return tonumber(x1), tonumber(y1), tonumber(x2), tonumber(y2)
        end
    end

    return nil
end

--- Strip old color/blur/alpha/clip overrides cleanly while keeping other tags and text intact.
local function strip_conflicting_tags(text, opts)
    local function clean_block(inner)
        -- Always remove the temporary probe \clip / \iclip
        inner = inner:gsub("\\i?clip%([^%)]*%)", "")

        if opts.match_primary then
            inner = inner:gsub("\\1c&H[0-9a-fA-F]+&?", "")
            inner = inner:gsub("\\c&H[0-9a-fA-F]+&?", "")
        end
        if opts.match_border then
            inner = inner:gsub("\\3c&H[0-9a-fA-F]+&?", "")
        end
        if opts.match_blur then
            inner = inner:gsub("\\blur%-?%d*%.?%d*", "")
            inner = inner:gsub("\\be%-?%d*%.?%d*", "")
        end
        if opts.match_grain then
            inner = inner:gsub("\\alpha&H[0-9a-fA-F]+&?", "")
        end

        return inner
    end

    local cleaned = text:gsub("{([^}]*)}", function(inner)
        local c = clean_block(inner)
        if c == "" then
            return ""
        end
        return "{" .. c .. "}"
    end)

    return cleaned
end

--- Inject the compiled ASS tags cleanly into the leading override block of the line.
local function apply_matched_tags(text, ass_tags, opts)
    local cleaned = strip_conflicting_tags(text, opts)
    if not ass_tags or ass_tags == "" then
        return cleaned
    end

    -- Extract inner tag content without outer braces
    local new_inner = ass_tags:match("^{(.*)}$") or ass_tags
    if new_inner == "" then
        return cleaned
    end

    -- If the line starts with an override block {...}, merge into it; otherwise prepend
    local existing_inner, rest = cleaned:match("^{([^}]*)}(.*)$")
    if existing_inner then
        return "{" .. new_inner .. existing_inner .. "}" .. rest
    else
        return "{" .. new_inner .. "}" .. cleaned
    end
end

-- ---------------------------------------------------------------------------
-- Dialog & Main Execution
-- ---------------------------------------------------------------------------
local function split_dir_file(filepath)
    if type(filepath) ~= "string" or filepath == "" then
        return get_script_dir(), "match_worker.py"
    end
    local dir, file = filepath:match("^(.*)[/\\]([^/\\]+)$")
    if dir and file then
        return (dir ~= "" and dir or "."), file
    end
    return get_script_dir(), filepath
end

local function show_dialog(initial_settings)
    local current = sanitize_settings(initial_settings)

    while true do
        local dialog_config = {
            { class = "label",    x = 0, y = 0, width = 1, height = 1, label = "Python Worker (.py):" },
            { class = "edit",     name = "py_path",       x = 1, y = 0, width = 2, height = 1, text = current.py_path },
            { class = "label",    x = 0, y = 1, width = 3, height = 1, label = "Target Properties:" },
            { class = "checkbox", name = "match_primary", x = 0, y = 2, width = 3, height = 1, label = "Primary Color (\\1c)", value = current.match_primary },
            { class = "checkbox", name = "match_border",  x = 0, y = 3, width = 3, height = 1, label = "Border Color (\\3c)",  value = current.match_border },
            { class = "checkbox", name = "match_blur",    x = 0, y = 4, width = 3, height = 1, label = "Edge Blur (\\blur)",    value = current.match_blur },
            { class = "checkbox", name = "match_grain",   x = 0, y = 5, width = 3, height = 1, label = "Grain Vector Mask",     value = current.match_grain },
            { class = "label",    x = 0, y = 6, width = 1, height = 1, label = "Sample Radius (px):" },
            { class = "intedit",  name = "radius",        x = 1, y = 6, width = 2, height = 1, value = current.radius, min = 1, max = 5 },
        }
        local buttons = { "Match Style", "Choose .py...", "Cancel" }
        local pressed, results = aegisub.dialog.display(dialog_config, buttons, { ok = "Match Style", cancel = "Cancel" })

        if pressed == "Choose .py..." then
            current = sanitize_settings(results)
            local default_dir, default_file = split_dir_file(current.py_path)
            local picked = aegisub.dialog.open(
                "Select Python Worker Script (match_worker.py)",
                default_file,
                default_dir,
                "Python Scripts (*.py)|*.py|All Files (*.*)|*.*",
                false,
                true
            )
            if picked and picked ~= "" then
                current.py_path = picked
                save_settings(current)
            end
        elseif pressed == "Match Style" then
            local final_opts = sanitize_settings(results)
            save_settings(final_opts)
            return final_opts
        else
            return nil
        end
    end
end

local function build_worker_command(params)
    local worker_path = params.py_path
    local py_exe = is_windows() and "python" or "python3"

    local args = {
        py_exe,
        quote_arg(worker_path),
        "--video", quote_arg(params.video_file),
        "--frame", tostring(params.frame),
        "--playresx", tostring(params.playresx),
        "--playresy", tostring(params.playresy),
        "--x1", tostring(params.x1),
        "--y1", tostring(params.y1),
        "--x2", tostring(params.x2),
        "--y2", tostring(params.y2),
        "--radius", tostring(params.radius),
        "--match-primary", params.match_primary and "1" or "0",
        "--match-border", params.match_border and "1" or "0",
        "--match-blur", params.match_blur and "1" or "0",
        "--match-grain", params.match_grain and "1" or "0",
    }

    local cmd = table.concat(args, " ")
    if is_windows() then
        -- Wrap entire command in quotes for Windows cmd /c via io.popen
        cmd = '"' .. cmd .. ' 2>&1"'
    else
        cmd = cmd .. " 2>&1"
    end
    return cmd
end

function run_style_matcher(subtitles, selected_lines, active_line)
    local settings = load_settings()

    local target_idx = active_line or selected_lines[1]
    if not target_idx or not subtitles[target_idx] then
        aegisub.dialog.display(
            { { class = "label", x = 0, y = 0, label = "Please select an active subtitle line." } },
            { "OK" }
        )
        return
    end

    local line = subtitles[target_idx]
    local x1, y1, x2, y2 = extract_probe_points(line.text)
    if not x1 then
        aegisub.dialog.display(
            { { class = "label", x = 0, y = 0, label = "No 2-point \\clip probe found on the active line.\nDraw \\clip(m x1 y1 l x2 y2) or \\clip(x1,y1,x2,y2) from text fill to background." } },
            { "OK" }
        )
        return
    end

    local props = aegisub.project_properties and aegisub.project_properties() or {}
    local video_file = props.video_file or ""
    if video_file == "" then
        aegisub.dialog.display(
            { { class = "label", x = 0, y = 0, label = "No video file is currently loaded in Aegisub." } },
            { "OK" }
        )
        return
    end

    local frame_idx = props.video_position
    if not frame_idx and aegisub.frame_from_ms then
        frame_idx = aegisub.frame_from_ms(line.start_time)
    end
    frame_idx = frame_idx or 0

    local opts = show_dialog(settings)
    if not opts then
        return
    end

    local py_check = io.open(opts.py_path, "r")
    if not py_check then
        aegisub.dialog.display(
            { { class = "label", x = 0, y = 0, label = "Python worker script not found:\n" .. tostring(opts.py_path) } },
            { "OK" }
        )
        return
    end
    py_check:close()

    local playresx, playresy = get_play_res(subtitles)

    local cmd = build_worker_command({
        py_path = opts.py_path,
        video_file = video_file,
        frame = frame_idx,
        playresx = playresx,
        playresy = playresy,
        x1 = x1,
        y1 = y1,
        x2 = x2,
        y2 = y2,
        radius = opts.radius or 2,
        match_primary = opts.match_primary,
        match_border = opts.match_border,
        match_blur = opts.match_blur,
        match_grain = opts.match_grain,
    })

    local pipe = io.popen(cmd, "r")
    if not pipe then
        aegisub.dialog.display(
            { { class = "label", x = 0, y = 0, label = "Failed to execute match_worker.py via io.popen()." } },
            { "OK" }
        )
        return
    end

    local raw_out = pipe:read("*a") or ""
    pipe:close()

    -- Extract JSON line from stdout
    local json_line = raw_out:match("({.*})")
    if not json_line then
        aegisub.dialog.display(
            { { class = "label", x = 0, y = 0, label = "Worker returned unexpected output:\n" .. raw_out } },
            { "OK" }
        )
        return
    end

    local ok, data = pcall(json_decode, json_line)
    if not ok or type(data) ~= "table" then
        aegisub.dialog.display(
            { { class = "label", x = 0, y = 0, label = "Failed to parse JSON output:\n" .. json_line } },
            { "OK" }
        )
        return
    end

    if not data.ok then
        aegisub.dialog.display(
            { { class = "label", x = 0, y = 0, label = "Style Matcher error:\n" .. tostring(data.error or "Unknown error") } },
            { "OK" }
        )
        return
    end

    line.text = apply_matched_tags(line.text, data.ass_tags or "", opts)
    subtitles[target_idx] = line

    -- If grain vector mask is enabled and detected, append a subtle grain vector overlay line
    if opts.match_grain and data.grain_detected and data.grain_vector and data.grain_vector ~= "" then
        local grain_line = {}
        for k, v in pairs(line) do
            grain_line[k] = v
        end
        grain_line.layer = (line.layer or 0) + 1
        grain_line.effect = "grain-mask"
        grain_line.text = data.grain_vector
        subtitles.insert(target_idx + 1, grain_line)
    end

    aegisub.set_undo_point(script_name)
end

local function validate_style_matcher(subtitles, selected_lines, active_line)
    local target_idx = active_line or selected_lines[1]
    return target_idx ~= nil and subtitles[target_idx] ~= nil
end

aegisub.register_macro(": RW Scripts :/" .. script_name, script_description, run_style_matcher, validate_style_matcher)
