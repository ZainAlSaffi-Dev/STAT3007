--[[
The filter for the G2 deck.

It does three jobs.

1. It labels each slide with the act it belongs to. A level 1 heading with the
   class "act" starts an act, and a level 1 heading with the class "backup"
   starts the backup slides. The filter numbers the acts and gives each act
   slide a dark background. It writes a small label with the act name and a
   track of all the acts at the top of every slide that follows.

2. It swaps each animation placeholder for its clip once the clip exists. A
   placeholder is a div with the class "anim" and a "scene" attribute that names
   the Manim scene. An optional "clip" attribute names the file when it differs
   from the scene, for example one section of a longer scene. The filter looks
   for the clip in assets/video and in animations/out/mp4. If there is no clip
   but there is a still, it shows the still. Otherwise it draws the div as an
   amber placeholder with its storyboard. A div with the class "figure-todo" is
   drawn as a placeholder in the same way.

3. It reads the time budget of each slide from its "time" attribute, written as
   minutes and seconds such as time="0:45". It writes the budget and the running
   totals into the speaker notes. When the deck renders, it prints the totals
   for each act and lists the clips that do not exist yet.
]]

-- The background of an act slide: the dark page with a thin band of the three
-- statistic colours along the bottom edge.
local NIGHT = "#121315"
local ACT_GRADIENT = "linear-gradient(to top, transparent 14px, #121315 14px), "
  .. "linear-gradient(90deg, #2a78d6 33.33%, #eb6834 33.33% 66.66%, #1baf7a 66.66%)"

-- Folders searched for a clip, best first, relative to the folder of the deck.
-- Manim names its quality folders by height and frame rate. A clip copied into
-- assets/video wins over all of them.
local VIDEO_DIRS = {
  "assets/video",
  "../animations/out/mp4/2160p60",
  "../animations/out/mp4/1440p60",
  "../animations/out/mp4/1080p60",
  "../animations/out/mp4/1080p30",
  "../animations/out/mp4",
  "../animations/out/mp4/720p30",
  "../animations/out/mp4/480p15",
}

-- Folders searched for a still. Manim may add its version to the file name, as in
-- KernelOnTheTorus_ManimCE_v0.19.0.png, so any name that starts with the scene
-- name and an underscore matches as well.
local STILL_DIRS = { "assets/stills", "../animations/out/stills" }

-- What the render log reports at the end.
local found = { anims = {}, figures = {} }

local function base_dir()
  if quarto and quarto.doc and quarto.doc.input_file then
    return pandoc.path.directory(quarto.doc.input_file)
  end
  return pandoc.system.get_working_directory()
end

local function exists(path)
  local handle = io.open(path, "r")
  if handle then
    handle:close()
    return true
  end
  return false
end

local function list_directory(path)
  local ok, entries = pcall(pandoc.system.list_directory, path)
  if ok and entries then
    return entries
  end
  return {}
end

local function escape_pattern(text)
  return (text:gsub("[%^%$%(%)%%%.%[%]%*%+%-%?]", "%%%0"))
end

local function escape_html(text)
  return (text:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"):gsub('"', "&quot;"))
end

local function find_video(clip)
  local base = base_dir()
  for _, dir in ipairs(VIDEO_DIRS) do
    local relative = dir .. "/" .. clip .. ".mp4"
    if exists(pandoc.path.join({ base, relative })) then
      return relative
    end
  end
  return nil
end

local function find_still(clip)
  local base = base_dir()
  local pattern = "^" .. escape_pattern(clip) .. "_ManimCE.*%.png$"
  for _, dir in ipairs(STILL_DIRS) do
    for _, name in ipairs(list_directory(pandoc.path.join({ base, dir }))) do
      if name == clip .. ".png" or name:match(pattern) then
        return dir .. "/" .. name
      end
    end
  end
  return nil
end

-- The amber box that stands in for a clip or a figure. The header line names the
-- item and shows its status, source and length. The storyboard written inside
-- the div stays below it.
local function placeholder(el, kind, name)
  local head = {
    '<div class="g2-ph-head">',
    '<span class="g2-ph-kind">' .. kind .. "</span>",
    '<span class="g2-ph-name">' .. escape_html(name) .. "</span>",
  }
  for _, key in ipairs({ "status", "source", "length" }) do
    local value = el.attributes[key]
    if value and value ~= "" then
      table.insert(head, '<span class="g2-chip">' .. escape_html(value) .. "</span>")
    end
  end
  table.insert(head, "</div>")

  local content = pandoc.Blocks({ pandoc.RawBlock("html", table.concat(head)) })
  content:extend(el.content)

  local classes = pandoc.List({ "g2-placeholder" })
  for _, class in ipairs(el.classes) do
    if class ~= "anim" and class ~= "figure-todo" then
      classes:insert(class)
    end
  end
  return pandoc.Div(content, pandoc.Attr(el.identifier, classes, {}))
end

local bar_chart

local function anim_or_figure(el)
  if el.classes:includes("g2-bars") then
    return bar_chart(el)
  end
  if el.classes:includes("anim") then
    local scene = el.attributes["scene"] or "Unnamed"
    local clip = el.attributes["clip"] or scene
    local video = find_video(clip)
    local still = find_still(clip)
    table.insert(found.anims, { name = clip, video = video, still = still })
    if video then
      local poster = still and (' poster="' .. still .. '"') or ""
      local loop = (el.attributes["loop"] == "true") and " loop" or ""
      return pandoc.RawBlock("html", string.format(
        '<video class="g2-video" data-autoplay muted playsinline preload="auto"%s%s src="%s"></video>',
        poster, loop, video))
    end
    if still then
      return pandoc.Para({ pandoc.Image({}, still, "", pandoc.Attr("", { "g2-still" })) })
    end
    return placeholder(el, "Animation", clip)
  end
  if el.classes:includes("figure-todo") then
    local name = el.attributes["name"] or "Unnamed"
    table.insert(found.figures, name)
    return placeholder(el, "Figure to make", name)
  end
  return nil
end

-- Bar charts. A div with the class "g2-bars" holds a list, and each item is one
-- bar written as a span: [label]{.is-rotation var="EtenRRsq"}. The value comes
-- from _variables.yml, which make_numbers.py writes from the report, so a chart
-- can never show a typed number. The class of the span picks the colour of the
-- bar: is-scale, is-rotation or is-alignment, and grey for anything else. The
-- div takes "max" for the end of the axis and "label" for the axis title. With
-- the class "stepwise", the bars appear one per click.
local variables = nil

local function load_variables()
  if variables then
    return variables
  end
  variables = {}
  local handle = io.open(pandoc.path.join({ base_dir(), "_variables.yml" }), "r")
  if not handle then
    return variables
  end
  for line in handle:lines() do
    local key, value = line:match('^([%w_]+):%s*"(.-)"')
    if key then
      variables[key] = value
    end
  end
  handle:close()
  return variables
end

local function to_number(text)
  if not text then
    return nil
  end
  local clean = text:gsub("\u{2212}", "-"):gsub(",", ""):gsub("^%+", "")
  return tonumber(clean)
end

local STATISTIC_CLASSES = { "is-scale", "is-rotation", "is-alignment" }

local function tick_step(max)
  for _, step in ipairs({ 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100 }) do
    if max / step <= 6 then
      return step
    end
  end
  return max / 5
end

function bar_chart(el)
  local list = nil
  for _, block in ipairs(el.content) do
    if block.t == "BulletList" then
      list = block
      break
    end
  end
  if not list then
    return nil
  end
  local vars = load_variables()

  -- Read every bar first, so the axis can fit the largest value.
  local bars = {}
  local largest = 0
  for _, item in ipairs(list.content) do
    local span = nil
    pandoc.walk_block(pandoc.Div(item), {
      Span = function(s)
        if not span and s.attributes["var"] then
          span = s
        end
      end,
    })
    if span then
      local name = span.attributes["var"]
      local text = vars[name]
      local value = to_number(text)
      if not value then
        io.stderr:write("[g2] Bar chart: no number for " .. name .. " in _variables.yml.\n")
      end
      local colour = "is-neutral"
      for _, class in ipairs(STATISTIC_CLASSES) do
        if span.classes:includes(class) then
          colour = class
        end
      end
      table.insert(bars, { label = span.content, text = text or "?", value = value or 0, colour = colour })
      largest = math.max(largest, value or 0)
    end
  end

  local max = tonumber(el.attributes["max"] or "") or largest * 1.15
  if max <= 0 then
    max = 1
  end
  local step = tick_step(max)
  local grid = string.format("--g2-grid: %.4f%%;", step / max * 100)
  local stepwise = el.classes:includes("stepwise")

  local rows = pandoc.Blocks({})
  for _, bar in ipairs(bars) do
    local width = math.max(0, math.min(bar.value, max)) / max * 100
    local track = pandoc.Span({
      pandoc.Span({}, pandoc.Attr("", { "g2-bar", bar.colour }, { style = string.format("width: %.2f%%;", width) })),
      pandoc.Span({ pandoc.Str(bar.text) }, pandoc.Attr("", { "g2-bar-value" },
        { style = string.format("left: %.2f%%;", width) })),
    }, pandoc.Attr("", { "g2-bar-track" }, { style = grid }))
    local classes = { "g2-bar-row" }
    if stepwise then
      table.insert(classes, "fragment")
    end
    rows:insert(pandoc.Div({
      pandoc.Plain({ pandoc.Span(bar.label, pandoc.Attr("", { "g2-bar-label" })), track }),
    }, pandoc.Attr("", classes, { title = pandoc.utils.stringify(bar.label) .. ": " .. bar.text })))
  end

  -- The axis: tick labels under the bars and a title below them.
  local ticks = {}
  local tick = 0
  while tick <= max + 1e-9 do
    local label = string.format("%g", math.floor(tick / step + 0.5) * step)
    table.insert(ticks, pandoc.Span({ pandoc.Str(label) }, pandoc.Attr("", { "g2-tick" },
      { style = string.format("left: %.2f%%;", tick / max * 100) })))
    tick = tick + step
  end
  rows:insert(pandoc.Div({
    pandoc.Plain({ pandoc.Span({}, pandoc.Attr("", { "g2-bar-label" })),
      pandoc.Span(ticks, pandoc.Attr("", { "g2-bar-track", "g2-axis-track" })) }),
  }, pandoc.Attr("", { "g2-bar-axis" })))
  if el.attributes["label"] then
    rows:insert(pandoc.Div({ pandoc.Plain({ pandoc.Str(el.attributes["label"]) }) },
      pandoc.Attr("", { "g2-bar-title" })))
  end

  local classes = pandoc.List({ "g2-barchart" })
  for _, class in ipairs(el.classes) do
    if class ~= "g2-bars" and class ~= "stepwise" then
      classes:insert(class)
    end
  end
  return pandoc.Div(rows, pandoc.Attr(el.identifier, classes, {}))
end

-- Time budgets.
local function seconds(text)
  if not text then
    return nil
  end
  local minutes, secs = text:match("^%s*(%d+):(%d%d)%s*$")
  if minutes then
    return tonumber(minutes) * 60 + tonumber(secs)
  end
  return nil
end

local function clock(total)
  return string.format("%d:%02d", total // 60, total % 60)
end

local function track_html(current, count)
  local bars = {}
  for i = 1, count do
    local class = ""
    if i < current then
      class = ' class="done"'
    elseif i == current then
      class = ' class="now"'
    end
    table.insert(bars, "<i" .. class .. "></i>")
  end
  return '<span class="g2-track">' .. table.concat(bars) .. "</span>"
end

local function kicker_html(label, number, count, optional)
  local parts = { '<div class="g2-kicker" aria-hidden="true">' }
  if number then
    table.insert(parts, '<span class="g2-kicker-num">' .. number .. "</span>")
  end
  table.insert(parts, "<span>" .. escape_html(label) .. "</span>")
  if optional then
    table.insert(parts, '<span class="g2-flag">optional</span>')
  end
  if number then
    table.insert(parts, track_html(number, count))
  end
  table.insert(parts, "</div>")
  return table.concat(parts)
end

local function structure(doc)
  local acts = {}
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 and block.classes:includes("act") then
      table.insert(acts, pandoc.utils.stringify(block.content))
    end
  end
  local count = #acts

  local target = seconds(pandoc.utils.stringify(doc.meta["g2-target"] or "10:00")) or 600
  local out = pandoc.Blocks({})
  local act = 0
  local backup = false
  local core, all = 0, 0
  local per_act = {}
  local pending_notes = nil
  local pending_track = nil

  -- Close the slide that is open: add its track, and write its budget into a
  -- new notes block if the slide had no notes of its own.
  local function close_slide()
    if pending_track then
      out:insert(pandoc.RawBlock("html", pending_track))
      pending_track = nil
    end
    if pending_notes then
      out:insert(pandoc.Div({ pending_notes }, pandoc.Attr("", { "notes" }, {})))
      pending_notes = nil
    end
  end

  local function budget(block, name)
    local t = seconds(block.attributes["time"])
    if not t or backup or block.attributes["visibility"] == "hidden" then
      return nil
    end
    local optional = block.classes:includes("optional")
    all = all + t
    if not optional then
      core = core + t
    end
    local key = act > 0 and acts[act] or "Opening"
    per_act[key] = per_act[key] or { core = 0, all = 0, order = act }
    per_act[key].all = per_act[key].all + t
    if not optional then
      per_act[key].core = per_act[key].core + t
    end
    local text = string.format(
      "Budget %s%s. Running total %s for the core slides and %s with the optional ones, against a target of %s.",
      clock(t), optional and " (optional slide)" or "", clock(core), clock(all), clock(target))
    return pandoc.Para({ pandoc.Emph({ pandoc.Str(text) }) })
  end

  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and (block.level == 1 or block.level == 2) then
      close_slide()
    end

    if block.t == "Header" and block.level == 1 then
      local name = pandoc.utils.stringify(block.content)
      if block.classes:includes("act") then
        act = act + 1
        backup = false
        block.classes:insert("g2-act")
        block.attributes["background-color"] = NIGHT
        block.attributes["background-gradient"] = ACT_GRADIENT
        block.attributes["menu-title"] = act .. ". " .. name
        block.content:insert(1, pandoc.RawInline("html",
          '<span class="g2-act-num">Part ' .. act .. " of " .. count .. "</span>"))
        pending_track = track_html(act, count)
      elseif block.classes:includes("backup") then
        backup = true
        block.attributes["visibility"] = "uncounted"
        block.attributes["background-color"] = NIGHT
        block.attributes["background-gradient"] = ACT_GRADIENT
        block.classes:insert("g2-act")
      end
      pending_notes = budget(block, name)
      out:insert(block)
    elseif block.t == "Header" and block.level == 2 then
      local optional = block.classes:includes("optional")
      if backup then
        block.attributes["visibility"] = "uncounted"
        block.classes:insert("g2-backup-slide")
      end
      pending_notes = budget(block, pandoc.utils.stringify(block.content))
      out:insert(block)
      if not block.classes:includes("g2-title-slide") then
        if backup then
          out:insert(pandoc.RawBlock("html", kicker_html("Backup", nil, count, false)))
        elseif act > 0 then
          out:insert(pandoc.RawBlock("html", kicker_html(acts[act], act, count, optional)))
        end
      end
    elseif block.t == "Div" and block.classes:includes("notes") and pending_notes then
      block.content:insert(1, pending_notes)
      pending_notes = nil
      out:insert(block)
    else
      out:insert(block)
    end
  end
  close_slide()
  doc.blocks = out

  -- The report in the render log.
  local log = {}
  table.insert(log, string.format("[g2] Time budget: %s for the core slides, %s with the optional ones, target %s.",
    clock(core), clock(all), clock(target)))
  local names = {}
  for key, value in pairs(per_act) do
    table.insert(names, { key = key, value = value })
  end
  table.sort(names, function(a, b) return a.value.order < b.value.order end)
  for _, item in ipairs(names) do
    table.insert(log, string.format("[g2]   %-22s core %s, all %s", item.key, clock(item.value.core), clock(item.value.all)))
  end
  local clips, stills = 0, 0
  local missing = {}
  for _, anim in ipairs(found.anims) do
    if anim.video then
      clips = clips + 1
    elseif anim.still then
      stills = stills + 1
    else
      table.insert(missing, anim.name)
    end
  end
  table.insert(log, string.format("[g2] Animations: %d placeholders, %d with a clip, %d with a still only.",
    #found.anims, clips, stills))
  if #missing > 0 then
    table.insert(log, "[g2]   No clip or still yet: " .. table.concat(missing, ", "))
  end
  if #found.figures > 0 then
    table.insert(log, "[g2] Figures still to make: " .. table.concat(found.figures, ", "))
  end
  io.stderr:write(table.concat(log, "\n") .. "\n")
  return doc
end

return {
  { Div = anim_or_figure },
  { Pandoc = structure },
}
