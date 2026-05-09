// nt_aimassist.mod.gml — per-player aim-assist for co-op.
//
// NT already implements aim-assist (Player/Step_0.gml:330) and the underlying
// KeyCont.aimassist[index] storage is per-player. The vanilla input handler
// just stamps the global opt_assist into every slot each frame
// (InputHandling.gml:226). This mod overrides that array per-player based on
// each player's input device, so the gamepad player gets assist while the
// KB/M player has clean mouse aim in the same co-op session.
//
// Behavior: ON for gamepad players, OFF for keyboard / mouse players.
// No in-game toggle — the device check is re-evaluated every step.
//
// (A "snap-and-stick" hard-lock variant was attempted but NTT's KeyCont
// schema is missing the dir_fire/dis_fire fields the rebuild has, so we can't
// override aim direction at the input layer from a mod. Run /gmlapi in-game
// to see what fields/functions NTT actually exposes if you want to extend.)
//
// Install: drop in NTT's mods folder, /loadmod nt_aimassist

#define init
trace("nt_aimassist loaded — assist ON for gamepad players, OFF for KB/M")

#define step
var _n = KeyCont.players
if (_n < 1) _n = 1
if (_n > 4) _n = 4

for (var _i = 0; _i < _n; _i += 1) {
    KeyCont.aimassist[_i] = KeyCont.gamepad[_i] ? 1 : 0
}

#define cleanup
trace("nt_aimassist cleanup")
