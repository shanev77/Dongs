import os, sys, json, math, time, random, threading
import pygame

from evo import Terrain, Food, Population, Toys
import sprites
from ai_llm import get_models, set_llm_target, llm_status_line, llm_rate_limits, pack_reply

# ---------- config ----------
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        CFG = json.load(f)
except Exception:
    CFG = {}

AIHUB_URL   = CFG.get("base_url", "http://192.168.0.10:11434")
DEFAULT_MOD = CFG.get("default_model", "granite3.1-moe:1b")

WORLD_W, WORLD_H = 1280, 720
UI_W = 540
SCREEN_W, SCREEN_H = WORLD_W + UI_W, WORLD_H

FPS_LIMIT   = 50
STEP_SLOW   = 0.85

llm_rate_limits(min_interval_sec=8.0, burst=1)

def ui_font(sz=18): return pygame.font.SysFont("consolas", sz)

def draw_button(surf, r, label, hot=False):
    bg = (40,52,62) if not hot else (54,84,108)
    pygame.draw.rect(surf, bg, r, border_radius=8)
    pygame.draw.rect(surf, (72,92,110), r, 2, border_radius=8)
    surf.blit(ui_font(18).render(label, True, (220,235,245)), (r.x+10, r.y+8))

def draw_entry(surf, r, text):
    pygame.draw.rect(surf, (24,28,36), r, border_radius=8)
    pygame.draw.rect(surf, (70,90,120), r, 2, border_radius=8)
    surf.blit(ui_font(18).render(text, True, (220,240,255)), (r.x+8, r.y+7))

def draw_dropdown(surf, r, items, idx, open_, enabled=True):
    border = (70,90,120) if enabled else (60,60,70)
    textc  = (220,240,255) if enabled else (150,160,170)
    base   = (24,28,36)
    pygame.draw.rect(surf, base, r, border_radius=8)
    pygame.draw.rect(surf, border, r, 2, border_radius=8)
    cur = items[idx] if items else "(no models)"
    surf.blit(ui_font(18).render(cur, True, textc), (r.x+8, r.y+7))
    pygame.draw.polygon(surf, textc,
                        [(r.right-20, r.y+12), (r.right-8, r.y+12), (r.right-14, r.y+20)])
    if open_ and items and enabled:
        lst = pygame.Rect(r.x, r.bottom+4, r.w, min(260, 26*len(items)+10))
        pygame.draw.rect(surf, (20,24,30), lst, border_radius=8)
        pygame.draw.rect(surf, border, lst, 2, border_radius=8)
        y = lst.y+6
        for i,it in enumerate(items):
            col = (140,200,255) if i==idx else (220,240,255)
            surf.blit(ui_font(18).render(it, True, col), (lst.x+8, y))
            y += 24
        return lst
    return None

def clamp(v, lo, hi): return lo if v<lo else (hi if v>hi else v)

class TermLine:
    USER="user"; PACK="pack"
    def __init__(self, text, kind):
        self.text=text; self.kind=kind

def main():
    pygame.init()
    pygame.display.set_caption("DONGS — AI Evolving Creatures")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()

    rng = random.Random()
    terrain = Terrain(WORLD_W, WORLD_H, cell=20, rng=rng)
    food    = Food(rng=rng, max_items=90, respawn_sec=4.0)
    toys    = Toys(rng=rng, count=10)
    pop     = Population(terrain, rng=rng, max_pop=100)

    world_static = pygame.Surface((WORLD_W, WORLD_H)).convert()
    sprites.draw_world(world_static, terrain)

    term_block = type("T", (), {})()
    term_block.x = terrain.gw*terrain.cell//2
    term_block.y = WORLD_H//2
    term_block.radius = 22
    term_block.occupied_by = None
    term_block.flash_until = 0.0

    # ---------- Right UI ----------
    PAD = 12
    ui_x  = WORLD_W + PAD
    top_y = PAD

    btn_start  = pygame.Rect(ui_x, top_y, 92, 34)
    btn_pause  = pygame.Rect(btn_start.right + 10, top_y, 92, 34)
    btn_stop   = pygame.Rect(btn_pause.right + 10, top_y, 92, 34)

    label_s = ui_font(18).render("Start:", True, (210,220,230))
    label_m = ui_font(18).render("Max:",   True, (210,220,230))

    entry_start = pygame.Rect(btn_stop.right + 20 + label_s.get_width(), top_y, 68, 34)
    entry_max   = pygame.Rect(entry_start.right + 28 + label_m.get_width(), top_y, 68, 34)

    start_count_txt, max_pop_txt = "25", "100"; which_edit = None

    dd_rect = pygame.Rect(ui_x, btn_start.bottom+12, UI_W - 2*PAD, 34)
    try:
        models = get_models(AIHUB_URL) or [DEFAULT_MOD]
    except Exception:
        models = [DEFAULT_MOD]
    selected_model = models.index(DEFAULT_MOD) if DEFAULT_MOD in models else 0
    dd_open = False
    set_llm_target(AIHUB_URL, models[selected_model])

    # Panels — shorten LLM panel ~half so dropdown fits comfortably
    term_h = 270
    insp_h = 130
    llm_h  = 150  # ↓ fixed shorter height
    term_panel = pygame.Rect(WORLD_W + PAD, SCREEN_H - PAD - term_h, UI_W-2*PAD, term_h)
    insp_panel = pygame.Rect(WORLD_W + PAD, term_panel.y - 10 - insp_h, UI_W-2*PAD, insp_h)
    llm_panel  = pygame.Rect(WORLD_W + PAD, insp_panel.y - 10 - llm_h, UI_W-2*PAD, llm_h)

    term_input = ""; term_has_focus = False
    term_lines: list[TermLine] = []
    term_scroll = 0

    running_sim = False
    paused = False
    nn_steps = 0

    drag_target = None
    drag_off = (0.0, 0.0)

    # logging state
    LOG_DIR = os.path.join(os.getcwd(), "Programs")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_rows = []
    log_file = None
    last_log_t = 0.0
    run_stamp = None

    def begin_log():
        nonlocal log_file, log_rows, last_log_t, run_stamp
        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(LOG_DIR, f"DONG-{run_stamp}.txt")
        log_rows = []
        last_log_t = 0.0
        header = [
            "timestamp","dong_id","age","energy","fitness",
            "gene_max_intel","gene_nn_growth","gene_pref_eat","gene_pref_play","gene_pref_breed","gene_explore_bias","gene_move_scalar"
        ]
        def labels(prefix, shape):
            if len(shape)==2:
                return [f"{prefix}_{i}_{j}" for i in range(shape[0]) for j in range(shape[1])]
            return [f"{prefix}_{i}" for i in range(shape[0])]
        header += labels("w1",(8,4)) + labels("b1",(8,)) + labels("w2",(2,8)) + labels("b2",(2,))
        log_rows.append("\t".join(header))

    def append_snapshot(tnow: float):
        for b in pop.bots:
            w1,b1,w2,b2 = b.brain
            row = [
                time.strftime("%H:%M:%S", time.localtime(tnow)),
                b.display_id,
                f"{b.age:.2f}", f"{b.energy:.2f}", f"{b.fitness:.3f}",
                f"{b.genes.max_intel:.3f}", f"{b.genes.nn_growth:.3f}",
                f"{b.genes.pref_eat:.3f}", f"{b.genes.pref_play:.3f}", f"{b.genes.pref_breed:.3f}",
                f"{b.genes.explore_bias:.3f}", f"{b.genes.move_scalar:.3f}"
            ]
            row += [f"{float(x):.4f}" for x in w1.flatten()]
            row += [f"{float(x):.4f}" for x in b1.flatten()]
            row += [f"{float(x):.4f}" for x in w2.flatten()]
            row += [f"{float(x):.4f}" for x in b2.flatten()]
            log_rows.append("\t".join(row))

    def write_log(final_note=""):
        if not log_file or not log_rows: return
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("# DONGS Simulation Log\n")
            f.write(f"# Started: {run_stamp}\n")
            f.write(f"# LLM base: {AIHUB_URL}\n")
            f.write(f"# Model: {models[selected_model]}\n")
            if final_note: f.write(f"# Note: {final_note}\n")
            f.write("\n".join(log_rows))

    def start_sim():
        nonlocal running_sim, paused, nn_steps, world_static, term_lines, term_scroll, last_log_t
        n  = clamp(int(start_count_txt or "1"), 1, 999)
        mx = clamp(int(max_pop_txt or "10"), 1, 999)
        pop.reset(max_pop=mx)
        pop.spawn(n)
        world_static = pygame.Surface((WORLD_W, WORLD_H)).convert()
        sprites.draw_world(world_static, terrain)
        running_sim = True; paused = False; nn_steps = 0
        term_lines.clear(); term_scroll = 0
        begin_log(); last_log_t = 0.0

    def stop_sim(final_note="stopped"):
        nonlocal running_sim, paused
        running_sim = False; paused=False
        write_log(final_note)

    def dd_list_rect():
        if not dd_open or not models: return None
        return pygame.Rect(dd_rect.x, dd_rect.bottom+4, dd_rect.w, min(260, 26*len(models)+10))

    def attract_curious_dong(_user_msg: str) -> bool:
        if term_block.occupied_by: return True
        candid = [(b, (b.x-term_block.x)**2 + (b.y-term_block.y)**2) for b in pop.bots]
        candid = [c for c in candid if c[1] < 220**2]
        if not candid: return False
        candid.sort(key=lambda t:t[1])
        for b,_ in candid[:8]:
            if rng.random() < 0.65:
                term_block.occupied_by = b
                b.waiting_for_reply = True
                b.docked_terminal = True
                b.fitness += 0.01
                return True
        return False

    def lines_fit():
        fh = ui_font(16).get_height()+2
        return max(3, (term_panel.h - 78)//fh)

    running = True
    while running:
        dt = clock.tick(FPS_LIMIT) / 1000.0
        if running_sim and not paused:
            dt *= STEP_SLOW
        else:
            dt = 0.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                if running_sim: write_log("quit")
                running = False

            elif ev.type == pygame.MOUSEWHEEL:
                if term_panel.collidepoint(pygame.mouse.get_pos()):
                    term_scroll = max(0, min(term_scroll - ev.y, max(0, len(term_lines) - lines_fit())))

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if btn_start.collidepoint(ev.pos): 
                    start_sim()
                elif btn_pause.collidepoint(ev.pos) and running_sim:
                    paused = not paused
                elif btn_stop.collidepoint(ev.pos):  
                    stop_sim()
                elif dd_rect.collidepoint(ev.pos) and (not running_sim):
                    dd_open = not dd_open
                else:
                    lst = dd_list_rect()
                    if lst and lst.collidepoint(ev.pos):
                        idx = (ev.pos[1] - (lst.y+6)) // 24
                        if 0 <= idx < len(models):
                            selected_model = idx
                            set_llm_target(AIHUB_URL, models[selected_model])
                        dd_open = False
                    else:
                        dd_open = False
                        if entry_start.collidepoint(ev.pos):
                            which_edit, term_has_focus = "start", False
                        elif entry_max.collidepoint(ev.pos):
                            which_edit, term_has_focus = "max", False
                        elif term_panel.collidepoint(ev.pos):
                            which_edit, term_has_focus = None, True
                        else:
                            which_edit, term_has_focus = None, False
                        if paused and 0 <= ev.pos[0] < WORLD_W and 0 <= ev.pos[1] < WORLD_H:
                            nearest=None; best=24**2
                            for b in pop.bots:
                                d=(b.x-ev.pos[0])**2+(b.y-ev.pos[1])**2
                                if d<best: nearest=b; best=d
                            if nearest:
                                drag_target = nearest
                                drag_off = (nearest.x-ev.pos[0], nearest.y-ev.pos[1])

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                drag_target = None

            elif ev.type == pygame.MOUSEMOTION:
                if paused and drag_target is not None:
                    nx = clamp(ev.pos[0]+drag_off[0], 6, WORLD_W-6)
                    ny = clamp(ev.pos[1]+drag_off[1], 6, WORLD_H-6)
                    drag_target.x, drag_target.y = nx, ny

            elif ev.type == pygame.KEYDOWN:
                if which_edit:
                    tgt = start_count_txt if which_edit=="start" else max_pop_txt
                    if ev.key == pygame.K_BACKSPACE: tgt = tgt[:-1]
                    elif ev.key in (pygame.K_RETURN, pygame.K_ESCAPE): which_edit=None
                    else:
                        ch = ev.unicode
                        if ch.isdigit() and len(tgt) < 4: tgt += ch
                    if which_edit=="start": start_count_txt = tgt
                    else: max_pop_txt = tgt

                elif term_has_focus:
                    if ev.key == pygame.K_ESCAPE:
                        term_input=""; term_has_focus=False
                    elif ev.key == pygame.K_BACKSPACE:
                        term_input = term_input[:-1]
                    elif ev.key == pygame.K_RETURN:
                        msg = term_input.strip(); term_input=""
                        if msg:
                            term_lines.append(TermLine(f"> {msg}", TermLine.USER))
                            term_block.flash_until = pygame.time.get_ticks()/1000.0 + 0.9
                            term_scroll = 0
                            def respond():
                                # pull a Dong if possible; otherwise still reply
                                if not term_block.occupied_by:
                                    attract_curious_dong(msg)
                                reply = pack_reply(msg)
                                term_lines.append(TermLine(f"PACK≫ {reply}", TermLine.PACK))
                                if term_block.occupied_by:
                                    d = term_block.occupied_by
                                    d.waiting_for_reply = False
                                    d.docked_terminal = False
                                    d.fitness += 0.03
                                    term_block.occupied_by = None
                            threading.Thread(target=respond, daemon=True).start()
                    else:
                        if ev.unicode and 32 <= ord(ev.unicode) < 127:
                            term_input += ev.unicode

        # world updates
        if running_sim and not paused:
            food.update()
            toys.update(dt)
            pop.step(dt, terrain, food, term_block, toys)
            nn_steps += 1

            # keep in-bounds & bounce off water edge
            for b in pop.bots:
                b.x = max(6, min(WORLD_W-6, b.x))
                b.y = max(6, min(WORLD_H-6, b.y))
                if terrain.is_water(b.x, b.y):
                    for _ in range(6):
                        ang = random.random()*math.tau
                        rx = b.x + math.cos(ang)*4
                        ry = b.y + math.sin(ang)*4
                        if not terrain.is_water(rx, ry):
                            b.x, b.y = rx, ry; break

            # log snapshot every ~2 seconds
            tnow = time.time()
            if tnow - last_log_t > 2.0:
                append_snapshot(tnow)
                last_log_t = tnow

            if running_sim and len(pop.bots) == 0:
                stop_sim("all creatures died")

        # ---- draw world ----
        screen.blit(world_static, (0,0))
        sprites.draw_toys(screen, toys)
        sprites.draw_terminal_block(screen, term_block)
        sprites.draw_food(screen, food)
        sprites.draw_breeding_curtains(screen, pop.breeding_sessions)
        for b in pop.bots:
            sprites.draw_dong(screen, b)
            if getattr(b, "bubble", None) and (time.time() - getattr(b, "last_bubble_time", 0) < 2.0):
                sprites.draw_bubble(screen, b.x, b.y, f"{b.bubble}")

        screen.blit(ui_font(20).render(f"Dongs: {len(pop.bots)}/{pop.max_pop}", True, (240,240,240)), (10,8))
        screen.blit(ui_font(20).render(f"NN steps: {nn_steps:,}", True, (210,220,210)), (10,34))

        pygame.draw.rect(screen, (16,20,26), (WORLD_W, 0, UI_W, SCREEN_H))

        draw_button(screen, btn_start, "Start", not running_sim)
        draw_button(screen, btn_pause, ("Resume" if paused else "Pause"), running_sim)
        draw_button(screen, btn_stop,  "Stop",  running_sim)
        screen.blit(ui_font(18).render("Start:", True, (210,220,230)),
                    (entry_start.x - ui_font(18).size("Start:")[0] - 8, entry_start.y+8))
        draw_entry(screen, entry_start, start_count_txt)
        screen.blit(ui_font(18).render("Max:", True, (210,220,230)),
                    (entry_max.x - ui_font(18).size("Max:")[0] - 8, entry_max.y+8))
        draw_entry(screen, entry_max, max_pop_txt)

        draw_dropdown(screen, dd_rect, models, selected_model, dd_open, enabled=(not running_sim))

        # LLM panel (shorter)
        pygame.draw.rect(screen, (20,24,30), llm_panel, border_radius=8)
        pygame.draw.rect(screen, (70,90,120), llm_panel, 2, border_radius=8)
        screen.blit(ui_font(18).render("LLM", True, (220,240,255)), (llm_panel.x+8, llm_panel.y+6))
        status = f"LLM @ {AIHUB_URL}\nmodel={models[selected_model]}\n" + llm_status_line()
        sx, sy = llm_panel.x+6, llm_panel.y+28
        for line in status.split("\n"):
            screen.blit(ui_font(16).render(line, True, (220,230,245)), (sx, sy))
            sy += ui_font(16).get_height()+2

        # INSPECTOR
        pygame.draw.rect(screen, (20,24,30), insp_panel, border_radius=8)
        pygame.draw.rect(screen, (70,90,120), insp_panel, 2, border_radius=8)
        screen.blit(ui_font(18).render("INSPECTOR", True, (220,240,255)), (insp_panel.x+8, insp_panel.y+6))
        mx,my = pygame.mouse.get_pos()
        hover_txt = "Move cursor over a Dong to inspect."
        if 0 <= mx < WORLD_W and 0 <= my < WORLD_H:
            nearest=None; best=26**2
            for b in pop.bots:
                d=(b.x-mx)**2+(b.y-my)**2
                if d<best: nearest=b; best=d
            if nearest and best < 26**2:
                w1,b1,w2,b2 = nearest.brain
                hover_txt = (
                    f"{nearest.display_id}\n"
                    f"Age: {nearest.age:.1f}/{nearest.max_age:.1f}\n"
                    f"Energy: {nearest.energy:.2f}  Fit: {nearest.fitness:.3f}\n"
                    f"Genes: maxInt={nearest.genes.max_intel:.2f} "
                    f"nnGrow={nearest.genes.nn_growth:.2f} eat={nearest.genes.pref_eat:.2f} "
                    f"play={nearest.genes.pref_play:.2f} breed={nearest.genes.pref_breed:.2f} "
                    f"explore={nearest.genes.explore_bias:.2f} move={nearest.genes.move_scalar:.2f}\n"
                    f"NN: w1 μ={float(w1.mean()):.2f} σ={float(w1.std()):.2f} | "
                    f"w2 μ={float(w2.mean()):.2f} σ={float(w2.std()):.2f}"
                )
        wrect = pygame.Rect(insp_panel.x+6, insp_panel.y+28, insp_panel.w-12, insp_panel.h-36)
        y = wrect.y
        for raw in hover_txt.split("\n"):
            screen.blit(ui_font(16).render(raw, True, (220,230,245)), (wrect.x, y))
            y += ui_font(16).get_height()+2

        # Terminal panel (chat only)
        term_panel_bg = (20,24,30)
        pygame.draw.rect(screen, term_panel_bg, term_panel, border_radius=8)
        pygame.draw.rect(screen, (70,90,120), term_panel, 2, border_radius=8)
        screen.blit(ui_font(18).render("TERMINAL", True, (220,240,255)), (term_panel.x+8, term_panel.y+6))
        N = lines_fit()
        if term_lines:
            start = max(0, len(term_lines) - N - term_scroll)
            end   = max(0, len(term_lines) - term_scroll)
            y = term_panel.y + 30
            for tl in term_lines[start:end]:
                col = (235,235,240) if tl.kind==TermLine.USER else (140,200,255)
                screen.blit(ui_font(16).render(tl.text, True, col), (term_panel.x+6, y))
                y += ui_font(16).get_height()+2

        input_rect = pygame.Rect(term_panel.x+8, term_panel.bottom-36, term_panel.w-16, 28)
        pygame.draw.rect(screen, (16,18,22), input_rect, border_radius=6)
        pygame.draw.rect(screen, (56,70,96), input_rect, 2, border_radius=6)
        screen.blit(ui_font(18).render("> " + term_input, True, (220,240,255)),
                    (input_rect.x+8, input_rect.y+4))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
        input("\n[Press Enter to close]")
