// nt_recorder.mod.gml — NTT mod that records human gameplay demonstrations
// for RL training. Outputs JSON-lines files compatible with the ntt_log_converter.
//
// Install: place in Nuclear Throne Together's mods folder and /load nt_recorder

#define init
// --- Episode & file tracking ---
global.rec_session_id = string(current_time) + "_" + string(irandom(9999));  // unique per launch
global.rec_episode    = 0;
global.rec_frame      = 0;
global.rec_buffer     = "";       // accumulated JSONL lines
global.rec_filename   = "";       // current episode file path
global.rec_active     = false;    // true while recording an episode
global.rec_flush_interval = 300;  // flush to disk every N frames
global.rec_frames_since_flush = 0;
global.rec_max_enemies = 20;

// --- Previous-frame state for reward signal deltas ---
global.rec_prev_kills    = 0;
global.rec_prev_hp       = 0;
global.rec_prev_area     = 0;
global.rec_prev_subarea  = 0;
global.rec_prev_level    = 0;

// --- Death tracking for episode boundary ---
global.rec_was_dead = true;  // start as "dead" so first player spawn starts an episode

trace("nt_recorder loaded — recording human demos");

#define step
// Only record when a player exists
if (!instance_exists(Player)) {
    // If we were recording and player no longer exists, the player died
    if (global.rec_active) {
        // Record one final frame marking death if possible
        rec_flush_episode();
        global.rec_active = false;
        global.rec_was_dead = true;
    }
    return;
}

var _p = instance_nearest(0, 0, Player);

// --- Detect episode boundaries ---
// New episode: player exists but we were dead or inactive
if (global.rec_was_dead || !global.rec_active) {
    // Start a new episode
    rec_start_episode();
    global.rec_was_dead = false;
}

// --- Check for player death mid-frame ---
if (_p.my_health <= 0) {
    // Record this death frame, then end episode
    rec_record_frame(_p);
    rec_flush_episode();
    global.rec_active = false;
    global.rec_was_dead = true;
    return;
}

// --- Normal frame recording ---
rec_record_frame(_p);

// --- Periodic flush ---
global.rec_frames_since_flush += 1;
if (global.rec_frames_since_flush >= global.rec_flush_interval) {
    rec_flush_partial();
}

#define draw
// Recording indicator
if (global.rec_active) {
    // Red dot in top-left corner
    draw_set_color(c_red);
    draw_set_alpha(0.9);
    draw_circle(view_xview[0] + 16, view_yview[0] + 16, 6, false);

    // Text info
    draw_set_color(c_white);
    draw_set_alpha(1.0);
    draw_set_font(-1);
    draw_text(view_xview[0] + 28, view_yview[0] + 9, "REC E" + string(global.rec_episode) + " F" + string(global.rec_frame));
}

#define cleanup
// Flush any remaining data
if (global.rec_active && global.rec_buffer != "") {
    rec_flush_partial();
}
trace("nt_recorder cleanup — flushed remaining data");

// ============================================================
// Helper functions
// ============================================================

#define rec_start_episode
// Finalize previous episode if somehow still active
if (global.rec_active && global.rec_buffer != "") {
    rec_flush_partial();
}

global.rec_episode += 1;
global.rec_frame = 0;
global.rec_buffer = "";
global.rec_frames_since_flush = 0;
global.rec_active = true;

// Generate filename with session ID + episode number (unique across sessions)
global.rec_filename = "ntt_demo_" + global.rec_session_id + "_" + string_zeros(global.rec_episode, 4) + ".jsonl";

// Initialize prev-frame trackers from current game state
if (instance_exists(GameCont)) {
    global.rec_prev_kills   = GameCont.kills;
    global.rec_prev_area    = GameCont.area;
    global.rec_prev_subarea = GameCont.subarea;
    global.rec_prev_level   = GameCont.level;
}

var _p = instance_nearest(0, 0, Player);
if (instance_exists(_p)) {
    global.rec_prev_hp = _p.my_health;
}

trace("nt_recorder — episode " + string(global.rec_episode) + " started -> " + global.rec_filename);

#define rec_record_frame(_p)
// --- Player state ---
var _px = _p.x;
var _py = _p.y;
var _php = _p.my_health;
var _pmaxhp = _p.maxhealth;
var _phsp = _p.hspeed;
var _pvsp = _p.vspeed;
var _pgun = _p.gunangle;
var _pwep = _p.wep;
var _pbwep = _p.bwep;
var _preload = _p.reload;
var _pcanshoot = _p.can_shoot;
var _proll = _p.roll;
var _prace = _p.race;
var _pnexthurt = _p.nexthurt;

// --- Wall raycast distances (4 cardinal directions) ---
// Step in 8px increments out to 300px and find the first wall hit.
// Mirrors scr_agent_build_state._agent_raycast_wall in the rebuild.
// Defaults to 1.0 (no wall) if collision_line / Wall isn't available.
var _wall_max = 300;
var _wall_step = 8;
var _wall_e = 1.0;
var _wall_n = 1.0;
var _wall_w = 1.0;
var _wall_s = 1.0;

if (object_exists(Wall)) {
    var _de = _wall_max;
    var _dn = _wall_max;
    var _dw = _wall_max;
    var _ds = _wall_max;
    var _found_e = false;
    var _found_n = false;
    var _found_w = false;
    var _found_s = false;
    var _d = _wall_step;
    while (_d <= _wall_max) {
        if (!_found_e && collision_line(_px, _py, _px + _d, _py, Wall, true, true) != noone) {
            _de = _d;
            _found_e = true;
        }
        if (!_found_n && collision_line(_px, _py, _px, _py - _d, Wall, true, true) != noone) {
            _dn = _d;
            _found_n = true;
        }
        if (!_found_w && collision_line(_px, _py, _px - _d, _py, Wall, true, true) != noone) {
            _dw = _d;
            _found_w = true;
        }
        if (!_found_s && collision_line(_px, _py, _px, _py + _d, Wall, true, true) != noone) {
            _ds = _d;
            _found_s = true;
        }
        if (_found_e && _found_n && _found_w && _found_s) break;
        _d += _wall_step;
    }
    _wall_e = min(_de / _wall_max, 1.0);
    _wall_n = min(_dn / _wall_max, 1.0);
    _wall_w = min(_dw / _wall_max, 1.0);
    _wall_s = min(_ds / _wall_max, 1.0);
}

// Ammo array - read each element
var _ammo0 = _p.ammo[0];
var _ammo1 = _p.ammo[1];
var _ammo2 = _p.ammo[2];
var _ammo3 = _p.ammo[3];
var _ammo4 = _p.ammo[4];
var _ammo5 = _p.ammo[5];

// --- Enemy collection ---
var _enemy_str = "";
var _enemy_count = 0;

if (instance_exists(enemy)) {
    // Collect enemies with distance
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

    // Insertion sort by distance (ascending) — simple, avoids NTT limitations
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

    // Cap at max enemies and build JSON string
    _enemy_count = min(_total, global.rec_max_enemies);
    for (var _i = 0; _i < _enemy_count; _i += 1) {
        if (_i > 0) _enemy_str += ",";
        _enemy_str += '{"x":' + string(_elist_x[_i])
            + ',"y":' + string(_elist_y[_i])
            + ',"my_health":' + string(_elist_hp[_i])
            + ',"maxhealth":' + string(_elist_maxhp[_i])
            + ',"type_id":' + string(_elist_type[_i]) + '}';
    }
}

// --- Enemy projectile collection (nearest 20, sorted by distance) ---
// EnemyBullet1 is the parent class for all enemy projectiles in NT/NTT
// (Guardian bullets, IDPD bullets, Throne2Ball, etc all inherit from it).
// Mirrors the projectile collection in scr_agent_build_state.
var _proj_str = "";
var _proj_count = 0;

if (object_exists(EnemyBullet1) && instance_exists(EnemyBullet1)) {
    var _pn = instance_number(EnemyBullet1);
    var _plist_x = array_create(_pn);
    var _plist_y = array_create(_pn);
    var _plist_hsp = array_create(_pn);
    var _plist_vsp = array_create(_pn);
    var _plist_dmg = array_create(_pn);
    var _plist_life = array_create(_pn);
    var _plist_dist = array_create(_pn);
    var _pi = 0;

    with (EnemyBullet1) {
        var _pdmg = variable_instance_exists(id, "damage") ? damage : 1;
        var _plife = variable_instance_exists(id, "lifetime") ? lifetime : 0;
        _plist_x[_pi] = x;
        _plist_y[_pi] = y;
        _plist_hsp[_pi] = hspeed;
        _plist_vsp[_pi] = vspeed;
        _plist_dmg[_pi] = _pdmg;
        _plist_life[_pi] = _plife;
        _plist_dist[_pi] = point_distance(x, y, _px, _py);
        _pi += 1;
    }

    var _ptotal = _pi;

    // Insertion sort by distance ascending (NTT lacks array_sort)
    var _pj, _pkx, _pky, _pkhsp, _pkvsp, _pkdmg, _pklife, _pkd;
    for (var _i = 1; _i < _ptotal; _i += 1) {
        _pkd = _plist_dist[_i];
        _pkx = _plist_x[_i];
        _pky = _plist_y[_i];
        _pkhsp = _plist_hsp[_i];
        _pkvsp = _plist_vsp[_i];
        _pkdmg = _plist_dmg[_i];
        _pklife = _plist_life[_i];
        _pj = _i - 1;
        while (_pj >= 0 && _plist_dist[_pj] > _pkd) {
            _plist_dist[_pj + 1] = _plist_dist[_pj];
            _plist_x[_pj + 1] = _plist_x[_pj];
            _plist_y[_pj + 1] = _plist_y[_pj];
            _plist_hsp[_pj + 1] = _plist_hsp[_pj];
            _plist_vsp[_pj + 1] = _plist_vsp[_pj];
            _plist_dmg[_pj + 1] = _plist_dmg[_pj];
            _plist_life[_pj + 1] = _plist_life[_pj];
            _pj -= 1;
        }
        _plist_dist[_pj + 1] = _pkd;
        _plist_x[_pj + 1] = _pkx;
        _plist_y[_pj + 1] = _pky;
        _plist_hsp[_pj + 1] = _pkhsp;
        _plist_vsp[_pj + 1] = _pkvsp;
        _plist_dmg[_pj + 1] = _pkdmg;
        _plist_life[_pj + 1] = _pklife;
    }

    // Cap at max projectiles and build JSON.
    // Emit pre-normalized values to match scr_agent_build_state's projectile output.
    _proj_count = min(_ptotal, global.rec_max_enemies);
    for (var _i = 0; _i < _proj_count; _i += 1) {
        if (_i > 0) _proj_str += ",";
        var _nx = _plist_x[_i] / 10080.0;
        var _ny = _plist_y[_i] / 10080.0;
        var _nhsp = max(-1.0, min(1.0, _plist_hsp[_i] / 10.0));
        var _nvsp = max(-1.0, min(1.0, _plist_vsp[_i] / 10.0));
        var _ndmg = min(_plist_dmg[_i] / 10.0, 1.0);
        var _nlife = min(_plist_life[_i] / 60.0, 1.0);
        _proj_str += '{"x":' + string(_nx)
            + ',"y":' + string(_ny)
            + ',"hspeed":' + string(_nhsp)
            + ',"vspeed":' + string(_nvsp)
            + ',"damage":' + string(_ndmg)
            + ',"lifetime":' + string(_nlife) + '}';
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

// --- Human input via NTT button API ---
var _btn_east  = button_check(0, "east");
var _btn_west  = button_check(0, "west");
var _btn_north = button_check(0, "north");
var _btn_south = button_check(0, "south");
var _btn_fire  = button_check(0, "fire");
var _btn_spec  = button_check(0, "spec");
var _btn_swap  = button_pressed(0, "swap");
var _btn_pick  = button_pressed(0, "pick");

// Derive movement direction from cardinal buttons
var _dx = _btn_east - _btn_west;
var _dy = _btn_south - _btn_north;
var _moving = (_dx != 0 || _dy != 0);
var _move_dir = 0;
if (_moving) {
    _move_dir = point_direction(0, 0, _dx, -_dy);
}

// Aim direction from player's gunangle
var _aim_dir = _pgun;

// --- Reward signals (raw deltas) ---
var _kills_delta = _kills - global.rec_prev_kills;
if (_kills_delta < 0) _kills_delta = 0;  // area transition can reset kills

var _damage_delta = global.rec_prev_hp - _php;
if (_damage_delta < 0) _damage_delta = 0;  // negative means healed, track separately

var _healed = false;
if (_php > global.rec_prev_hp && global.rec_frame > 0) {
    _healed = true;
}

var _level_changed = false;
if (_area != global.rec_prev_area || _subarea != global.rec_prev_subarea) {
    _level_changed = true;
}

// --- Build JSON string manually (NTT has no json_stringify) ---
var _json = '{"frame":' + string(global.rec_frame);

// Player object
_json += ',"player":{'
    + '"x":' + string(_px)
    + ',"y":' + string(_py)
    + ',"my_health":' + string(_php)
    + ',"maxhealth":' + string(_pmaxhp)
    + ',"hspeed":' + string(_phsp)
    + ',"vspeed":' + string(_pvsp)
    + ',"gunangle":' + string(_pgun)
    + ',"wep":' + string(_pwep)
    + ',"bwep":' + string(_pbwep)
    + ',"ammo":[' + string(_ammo0) + ',' + string(_ammo1) + ',' + string(_ammo2) + ',' + string(_ammo3) + ',' + string(_ammo4) + ',' + string(_ammo5) + ']'
    + ',"reload":' + string(_preload)
    + ',"can_shoot":' + rec_bool_str(_pcanshoot)
    + ',"roll":' + rec_bool_str(_proll)
    + ',"race":' + string(_prace)
    + ',"nexthurt":' + string(_pnexthurt)
    + ',"wall_dist_e":' + string(_wall_e)
    + ',"wall_dist_n":' + string(_wall_n)
    + ',"wall_dist_w":' + string(_wall_w)
    + ',"wall_dist_s":' + string(_wall_s)
    + '}';

// Enemies array
_json += ',"enemies":[' + _enemy_str + ']';

// Projectiles array (enemy bullets, nearest 20)
_json += ',"projectiles":[' + _proj_str + ']';

// Game state
_json += ',"game":{'
    + '"area":' + string(_area)
    + ',"subarea":' + string(_subarea)
    + ',"level":' + string(_level)
    + ',"loops":' + string(_loops)
    + ',"kills":' + string(_kills)
    + ',"hard":' + string(_hard)
    + '}';

// Human action
_json += ',"human_action":{'
    + '"move_dir":' + string(_move_dir)
    + ',"moving":' + rec_bool_str(_moving)
    + ',"aim_dir":' + string(_aim_dir)
    + ',"fire":' + rec_bool_str(_btn_fire)
    + ',"spec":' + rec_bool_str(_btn_spec)
    + ',"swap":' + rec_bool_str(_btn_swap)
    + ',"pick":' + rec_bool_str(_btn_pick)
    + '}';

// Reward signals
_json += ',"reward_signals":{'
    + '"kills_this_frame":' + string(_kills_delta)
    + ',"damage_this_frame":' + string(_damage_delta)
    + ',"healed_this_frame":' + rec_bool_str(_healed)
    + ',"level_changed":' + rec_bool_str(_level_changed)
    + '}';

_json += '}';

// Append to buffer
global.rec_buffer += _json + chr(10);

// Update prev-frame trackers
global.rec_prev_kills   = _kills;
global.rec_prev_hp      = _php;
global.rec_prev_area    = _area;
global.rec_prev_subarea = _subarea;
global.rec_prev_level   = _level;

global.rec_frame += 1;

#define rec_flush_episode
// Flush all buffered data as end-of-episode
if (global.rec_buffer != "" && global.rec_filename != "") {
    // Append to existing file content (in case of periodic flushes)
    var _existing = "";
    if (file_exists(global.rec_filename)) {
        _existing = string_load(global.rec_filename);
    }
    string_save(_existing + global.rec_buffer, global.rec_filename);
    trace("nt_recorder — episode " + string(global.rec_episode) + " saved (" + string(global.rec_frame) + " frames) -> " + global.rec_filename);
}
global.rec_buffer = "";
global.rec_frames_since_flush = 0;

#define rec_flush_partial
// Safety flush — write accumulated frames to disk without ending episode
if (global.rec_buffer != "" && global.rec_filename != "") {
    var _existing = "";
    if (file_exists(global.rec_filename)) {
        _existing = string_load(global.rec_filename);
    }
    string_save(_existing + global.rec_buffer, global.rec_filename);
}
global.rec_buffer = "";
global.rec_frames_since_flush = 0;

#define rec_bool_str(_val)
// NTT GML has no native JSON bool; output "true"/"false" strings for JSON
if (_val) {
    return "true";
}
return "false";

#define string_zeros(_val, _digits)
// Pad a number with leading zeros to _digits width
var _s = string(_val);
while (string_length(_s) < _digits) {
    _s = "0" + _s;
}
return _s;
