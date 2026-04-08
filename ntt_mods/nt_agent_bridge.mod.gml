// nt_agent_bridge.mod.gml — NTT mod that bridges a trained RL agent to
// the official Nuclear Throne game via file-based IPC.
//
// A Python adapter (ntt_bridge_adapter.py) sits between this mod and the
// NuclearThroneEnv TCP socket, translating file reads/writes to TCP messages.
//
// Install: place in Nuclear Throne Together's mods folder and /load nt_agent_bridge

#define init
// --- IPC file paths ---
global.ipc_state_file   = "agent_state.json";
global.ipc_state_ready  = "agent_state.ready";
global.ipc_action_file  = "agent_action.json";
global.ipc_action_ready = "agent_action.ready";

// --- Agent state tracking ---
global.agent_mode  = true;
global.agent_frame = 0;
global.agent_connected = false;  // true once first action received
global.agent_max_enemies = 20;

// --- Previous-frame state for reward deltas ---
global.agent_prev_kills   = 0;
global.agent_prev_hp      = 0;
global.agent_prev_area    = 0;
global.agent_prev_subarea = 0;

// --- Previous action state for press/release detection ---
global.agent_prev_fire = false;
global.agent_prev_spec = false;

// --- Episode management ---
global.agent_done       = false;
global.agent_reset_pending = false;

// --- Clean up stale IPC files from previous sessions ---
if (file_exists(global.ipc_state_file))  file_delete(global.ipc_state_file);
if (file_exists(global.ipc_state_ready)) file_delete(global.ipc_state_ready);
if (file_exists(global.ipc_action_file)) file_delete(global.ipc_action_file);
if (file_exists(global.ipc_action_ready)) file_delete(global.ipc_action_ready);

trace("nt_agent_bridge loaded — waiting for adapter connection");

#define step
// --- Handle reset pending (from death screen) ---
if (global.agent_reset_pending) {
    // Check for new action/reset command
    if (file_exists(global.ipc_action_ready)) {
        var _raw = string_load(global.ipc_action_file);
        file_delete(global.ipc_action_file);
        file_delete(global.ipc_action_ready);

        // Check if it's a reset command
        // NTT doesn't have json_decode, so look for "reset" in the string
        if (string_pos('"reset"', _raw) > 0) {
            global.agent_reset_pending = false;
            global.agent_frame = 0;
            global.agent_done = false;
            global.agent_prev_fire = false;
            global.agent_prev_spec = false;
            game_restart();
            return;
        }
    }
    return;
}

// --- Only operate when a player exists ---
if (!instance_exists(Player)) {
    // If we were tracking a live player, they just died
    if (global.agent_prev_hp > 0 && global.agent_frame > 0) {
        // Send final death state
        agent_write_state(true);
        global.agent_done = true;
        global.agent_reset_pending = true;
        global.agent_prev_hp = 0;
    }
    return;
}

var _p = instance_nearest(0, 0, Player);

// --- Initialize trackers on first frame of episode ---
if (global.agent_frame == 0) {
    global.agent_prev_hp = _p.my_health;
    if (instance_exists(GameCont)) {
        global.agent_prev_kills   = GameCont.kills;
        global.agent_prev_area    = GameCont.area;
        global.agent_prev_subarea = GameCont.subarea;
    }
}

// --- Check for player death mid-frame ---
if (_p.my_health <= 0) {
    agent_write_state(true);
    global.agent_done = true;
    global.agent_reset_pending = true;
    global.agent_prev_hp = 0;
    return;
}

// --- Read action from adapter (if available) ---
if (file_exists(global.ipc_action_ready)) {
    var _raw = string_load(global.ipc_action_file);
    file_delete(global.ipc_action_file);
    file_delete(global.ipc_action_ready);

    if (!global.agent_connected) {
        global.agent_connected = true;
        trace("nt_agent_bridge — adapter connected (first action received)");
    }

    // Check for reset command
    if (string_pos('"reset"', _raw) > 0) {
        global.agent_reset_pending = false;
        global.agent_frame = 0;
        global.agent_done = false;
        global.agent_prev_fire = false;
        global.agent_prev_spec = false;
        game_restart();
        return;
    }

    // Parse action JSON — extract fields manually
    var _move_dir = agent_parse_number(_raw, '"move_dir"');
    var _moving   = agent_parse_bool(_raw, '"moving"');
    var _aim_dir  = agent_parse_number(_raw, '"aim_dir"');
    var _shoot    = agent_parse_bool(_raw, '"fire"');
    var _spec     = agent_parse_bool(_raw, '"spec"');

    // Apply movement via NTT button API
    // move_dir is an angle in degrees (0=E, 45=NE, 90=N, etc.)
    agent_apply_movement(_move_dir, _moving);

    // Apply aim — set gunangle directly
    _p.gunangle = _aim_dir;

    // Apply fire button
    button_set(0, "fire", _shoot);

    // Apply special button
    button_set(0, "spec", _spec);

    // Store for press/release detection
    global.agent_prev_fire = _shoot;
    global.agent_prev_spec = _spec;
}

// --- Write current state for the adapter ---
agent_write_state(false);

global.agent_frame += 1;

#define draw
// --- HUD indicator ---
draw_set_color(c_lime);
draw_set_alpha(0.9);
draw_circle(view_xview[0] + 16, view_yview[0] + 16, 6, false);

draw_set_color(c_white);
draw_set_alpha(1.0);
draw_set_font(-1);

var _status = "WAITING";
if (global.agent_connected) _status = "CONNECTED";
if (global.agent_done)      _status = "DONE";

draw_text(view_xview[0] + 28, view_yview[0] + 9,
    "AGENT BRIDGE [" + _status + "] F" + string(global.agent_frame));

#define cleanup
// --- Delete IPC files ---
if (file_exists(global.ipc_state_file))  file_delete(global.ipc_state_file);
if (file_exists(global.ipc_state_ready)) file_delete(global.ipc_state_ready);
if (file_exists(global.ipc_action_file)) file_delete(global.ipc_action_file);
if (file_exists(global.ipc_action_ready)) file_delete(global.ipc_action_ready);
trace("nt_agent_bridge cleanup — IPC files removed");

// ============================================================
// Helper functions
// ============================================================

#define agent_write_state(_done)
// Collect state, compute reward, build JSON, write to IPC files.

var _p = -1;
var _px = 0;
var _py = 0;
var _php = 0;
var _pmaxhp = 0;
var _phsp = 0;
var _pvsp = 0;
var _pgun = 0;
var _pwep = 0;
var _pbwep = 0;
var _preload = 0;
var _pcanshoot = false;
var _proll = false;
var _ammo0 = 0;
var _ammo1 = 0;
var _ammo2 = 0;
var _ammo3 = 0;
var _ammo4 = 0;
var _ammo5 = 0;

if (instance_exists(Player)) {
    _p = instance_nearest(0, 0, Player);
    _px = _p.x;
    _py = _p.y;
    _php = _p.my_health;
    _pmaxhp = _p.maxhealth;
    _phsp = _p.hspeed;
    _pvsp = _p.vspeed;
    _pgun = _p.gunangle;
    _pwep = _p.wep;
    _pbwep = _p.bwep;
    _preload = _p.reload;
    _pcanshoot = _p.can_shoot;
    _proll = _p.roll;
    _ammo0 = _p.ammo[0];
    _ammo1 = _p.ammo[1];
    _ammo2 = _p.ammo[2];
    _ammo3 = _p.ammo[3];
    _ammo4 = _p.ammo[4];
    _ammo5 = _p.ammo[5];
}

// --- Enemy collection (nearest 20, sorted by distance) ---
var _enemy_str = "";
var _enemy_count = 0;

if (instance_exists(Player) && instance_exists(enemy)) {
    var _elist_x = array_create(instance_number(enemy));
    var _elist_y = array_create(instance_number(enemy));
    var _elist_hp = array_create(instance_number(enemy));
    var _elist_maxhp = array_create(instance_number(enemy));
    var _elist_type = array_create(instance_number(enemy));
    var _elist_dist = array_create(instance_number(enemy));
    var _ei = 0;

    with (enemy) {
        var _d = point_distance(x, y, _px, _py);
        _elist_x[_ei] = x;
        _elist_y[_ei] = y;
        _elist_hp[_ei] = my_health;
        _elist_maxhp[_ei] = maxhealth;
        _elist_type[_ei] = object_index;
        _elist_dist[_ei] = _d;
        _ei += 1;
    }

    var _total = _ei;

    // Insertion sort by distance ascending
    var _j, _kx, _ky, _khp, _kmhp, _kt, _kd;
    for (var _i = 1; _i < _total; _i += 1) {
        _kd = _elist_dist[_i];
        _kx = _elist_x[_i];
        _ky = _elist_y[_i];
        _khp = _elist_hp[_i];
        _kmhp = _elist_maxhp[_i];
        _kt = _elist_type[_i];
        _j = _i - 1;
        while (_j >= 0 && _elist_dist[_j] > _kd) {
            _elist_dist[_j + 1] = _elist_dist[_j];
            _elist_x[_j + 1] = _elist_x[_j];
            _elist_y[_j + 1] = _elist_y[_j];
            _elist_hp[_j + 1] = _elist_hp[_j];
            _elist_maxhp[_j + 1] = _elist_maxhp[_j];
            _elist_type[_j + 1] = _elist_type[_j];
            _j -= 1;
        }
        _elist_dist[_j + 1] = _kd;
        _elist_x[_j + 1] = _kx;
        _elist_y[_j + 1] = _ky;
        _elist_hp[_j + 1] = _khp;
        _elist_maxhp[_j + 1] = _kmhp;
        _elist_type[_j + 1] = _kt;
    }

    _enemy_count = min(_total, global.agent_max_enemies);
    for (var _i = 0; _i < _enemy_count; _i += 1) {
        if (_i > 0) _enemy_str += ",";
        _enemy_str += '{"x":' + string(_elist_x[_i])
            + ',"y":' + string(_elist_y[_i])
            + ',"hp":' + string(_elist_hp[_i])
            + ',"max_hp":' + string(_elist_maxhp[_i])
            + ',"hitid":' + string(_elist_type[_i]) + '}';
    }
}

// --- Game state ---
var _area = 0;
var _subarea = 0;
var _level = 0;
var _loops = 0;
var _kills = 0;
var _hard = 0;

if (instance_exists(GameCont)) {
    _area    = GameCont.area;
    _subarea = GameCont.subarea;
    _level   = GameCont.level;
    _loops   = GameCont.loops;
    _kills   = GameCont.kills;
    _hard    = GameCont.hard;
}

// --- Compute reward (delta-based, matches scr_agent_compute_reward) ---
var _reward = 0.01;  // survival bonus

// Kill reward
var _kills_delta = _kills - global.agent_prev_kills;
if (_kills_delta < 0) _kills_delta = 0;
_reward += _kills_delta * 5.0;

// Level/area change reward
if (_area != global.agent_prev_area || _subarea != global.agent_prev_subarea) {
    _reward += 10.0;
}

// Damage taken penalty
var _damage_taken = global.agent_prev_hp - _php;
if (_damage_taken > 0) {
    _reward += _damage_taken * -1.0;
}

// Heal reward (only when HP below 50%)
if (_php > global.agent_prev_hp && global.agent_frame > 0) {
    if (_pmaxhp > 0 && _php < _pmaxhp * 0.5) {
        _reward += 2.0;
    }
}

// Death penalty
if (_done) {
    _reward += -15.0;
}

// --- Build JSON string (NTT has no json_stringify) ---
var _json = '{"type":"state"';
_json += ',"frame":' + string(global.agent_frame);
_json += ',"done":' + agent_bool_str(_done);
_json += ',"reward":' + string(_reward);

// Player object — use rebuild variable names (hp, max_hp) for NuclearThroneEnv compatibility
_json += ',"player":{'
    + '"x":' + string(_px)
    + ',"y":' + string(_py)
    + ',"hp":' + string(_php)
    + ',"max_hp":' + string(_pmaxhp)
    + ',"hspeed":' + string(_phsp)
    + ',"vspeed":' + string(_pvsp)
    + ',"gunangle":' + string(_pgun)
    + ',"wep":' + string(_pwep)
    + ',"bwep":' + string(_pbwep)
    + ',"ammo":[' + string(_ammo0) + ',' + string(_ammo1) + ',' + string(_ammo2) + ',' + string(_ammo3) + ',' + string(_ammo4) + ',' + string(_ammo5) + ']'
    + ',"reload":' + string(_preload)
    + ',"can_shoot":' + agent_bool_str(_pcanshoot)
    + ',"roll":' + agent_bool_str(_proll)
    + '}';

// Enemies array
_json += ',"enemies":[' + _enemy_str + ']';

// Game state
_json += ',"game":{'
    + '"area":' + string(_area)
    + ',"subarea":' + string(_subarea)
    + ',"level":' + string(_level)
    + ',"loops":' + string(_loops)
    + ',"kills":' + string(_kills)
    + ',"hard":' + string(_hard)
    + '}';

_json += ',"mutation_screen":false';
_json += '}';

// --- Write to IPC files (data first, then sentinel) ---
string_save(_json, global.ipc_state_file);
string_save("1", global.ipc_state_ready);

// --- Update previous-frame trackers ---
global.agent_prev_kills   = _kills;
global.agent_prev_hp      = _php;
global.agent_prev_area    = _area;
global.agent_prev_subarea = _subarea;

#define agent_apply_movement(_move_dir, _moving)
// Decompose move_dir angle into cardinal button presses.
// move_dir is in degrees (GameMaker convention): 0=E, 90=N, 180=W, 270=S
// _moving is a boolean — if false, release all movement buttons.
var _east  = 0;
var _west  = 0;
var _north = 0;
var _south = 0;

if (_moving) {
    // Use cos/sin to decompose angle into cardinal directions,
    // matching scr_agent_apply_action's approach with lengthdir.
    var _dx = lengthdir_x(1, _move_dir);
    var _dy = lengthdir_y(1, _move_dir);
    _east  = (_dx > 0.3) ? 1 : 0;
    _west  = (_dx < -0.3) ? 1 : 0;
    _north = (_dy < -0.3) ? 1 : 0;  // GameMaker Y is inverted: negative = up
    _south = (_dy > 0.3) ? 1 : 0;
}

button_set(0, "north", _north);
button_set(0, "south", _south);
button_set(0, "east",  _east);
button_set(0, "west",  _west);

#define agent_parse_number(_str, _key)
// Extract a numeric value after a JSON key from a raw JSON string.
// e.g. agent_parse_number('{"move_dir":3,"aim_dir":45}', '"move_dir"') -> 3
var _pos = string_pos(_key, _str);
if (_pos <= 0) return 0;

// Skip past key and colon
_pos += string_length(_key);
var _rest = string_delete(_str, 1, _pos);

// Find the colon
var _colon = string_pos(":", _rest);
if (_colon <= 0) return 0;
_rest = string_delete(_rest, 1, _colon);

// Read until comma, closing brace, or end
var _num_str = "";
for (var _i = 1; _i <= string_length(_rest); _i += 1) {
    var _ch = string_char_at(_rest, _i);
    if (_ch == "," || _ch == "}" || _ch == "]" || _ch == " ") break;
    _num_str += _ch;
}

return real(_num_str);

#define agent_parse_bool(_str, _key)
// Extract a boolean value after a JSON key from a raw JSON string.
var _pos = string_pos(_key, _str);
if (_pos <= 0) return false;

_pos += string_length(_key);
var _rest = string_delete(_str, 1, _pos);

var _colon = string_pos(":", _rest);
if (_colon <= 0) return false;
_rest = string_delete(_rest, 1, _colon);

// Trim leading whitespace
while (string_length(_rest) > 0 && string_char_at(_rest, 1) == " ") {
    _rest = string_delete(_rest, 1, 1);
}

if (string_pos("true", _rest) == 1) return true;
return false;

#define agent_bool_str(_val)
// Output "true"/"false" strings for JSON
if (_val) {
    return "true";
}
return "false";
