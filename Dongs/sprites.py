import pygame, math, random

BG = (22,26,28)
GRASS1=(36,78,54); GRASS2=(40,86,58)
GRASS_SP1=(66,136,88); GRASS_SP2=(34,84,56)
WATER       = (72,132,230)
WATER_EDGE  = (26,50,100)
WATER_RING  = (98,156,240)

TREE_TRUNK=(84,62,40); TREE_SH=(18,26,30)
TREE_LEAF1=(38,120,52); TREE_LEAF2=(26,96,44)

ROCK=(100,110,120); ROCK_HI=(180,190,200); ROCK_SH=(50,60,70)

CRT_CASE     = (28, 32, 38)
CRT_EDGE     = (70, 90, 120)
CRT_SCREEN   = (18, 40, 50)
CRT_STAND    = (26, 30, 36)

DONG_SKIN       = (246,224,130)
DONG_SKIN_OLD   = (210,190,115)
DONG_BODY       = (52,120,200)
DONG_BODY_OLD   = (46,100,165)
DONG_WAIT       = (255,230,90)

TOY_BALL_1 = (95,150,255)
TOY_BALL_2 = (80,120,210)
TOY_STRIPE = (240,240,255)
TOY_BLOCK  = (220,190,80)
TOY_EDGE   = (30,40,60)

def draw_world(screen: pygame.Surface, terrain):
    pygame.draw.rect(screen, BG, (0,0,terrain.w, terrain.h))
    for ix in range(terrain.gw):
        for iy in range(terrain.gh):
            col = GRASS1 if (ix+iy)%2==0 else GRASS2
            r = pygame.Rect(ix*terrain.cell, iy*terrain.cell, terrain.cell, terrain.cell)
            pygame.draw.rect(screen, col, r)
            sp = float(terrain.speed[ix,iy])
            if 0.0 < sp < 0.95:
                pygame.draw.rect(screen, (34,72,56), r, 0)

    for cx,cy,r in terrain.water:
        x = cx*terrain.cell + terrain.cell//2
        y = cy*terrain.cell + terrain.cell//2
        rad = r*terrain.cell//2
        pygame.draw.circle(screen, WATER_EDGE, (x,y), rad + 5)
        pygame.draw.circle(screen, WATER,      (x,y), rad)
        pygame.draw.circle(screen, WATER_RING, (x,y), max(6, rad-6), width=2)

    for x,y in terrain.speckles:
        screen.fill(random.choice((GRASS_SP1, GRASS_SP2)), rect=(x,y,2,2))

    for x,y in terrain.trees:
        pygame.draw.ellipse(screen, (18,26,30), (x-18, y-10, 36, 16))
        pygame.draw.rect(screen, TREE_TRUNK, (x-3, y-4, 6, 18), border_radius=2)
        pygame.draw.circle(screen, TREE_LEAF2, (x, y-10), 18)
        pygame.draw.circle(screen, TREE_LEAF1, (x-6, y-12), 14)

    for x,y in terrain.rocks:
        pygame.draw.ellipse(screen, (50,60,70), (x-10, y+2, 20, 8))
        pygame.draw.circle(screen, (100,110,120), (x, y), 9)
        pygame.draw.circle(screen, (180,190,200), (x-3, y-2), 3)

def draw_food(screen, food):
    for x,y,_ in food.items:
        pygame.draw.circle(screen, (190,240,175), (int(x), int(y)), 4)

def _dong_palette_by_age(age: float, max_age: float):
    t = min(1.0, max(0.0, age/max_age))
    def lerp(a,b,t): return tuple(int(a[i]+(b[i]-a[i])*t) for i in range(3))
    skin = lerp(DONG_SKIN, DONG_SKIN_OLD, t*0.8)
    body = lerp(DONG_BODY, DONG_BODY_OLD, t*0.8)
    return skin, body

def draw_dong(screen, dong):
    x, y = int(dong.x), int(dong.y)
    t = pygame.time.get_ticks()/1000.0
    speed = (getattr(dong,'vx',0.0)**2 + getattr(dong,'vy',0.0)**2) ** 0.5
    walk_phase = (t*10.0 + (dong.id%5)*0.37) % 1.0
    stepping = speed > 8.0
    bob = (1.0 if stepping else 0.3)*math.sin(walk_phase*2*math.pi)
    skin, body = _dong_palette_by_age(dong.age, dong.max_age)

    if getattr(dong, "waiting_for_reply", False) or getattr(dong, "is_praying", False):
        pygame.draw.circle(screen, DONG_WAIT, (x,y), 13, width=2)

    # shadow
    pygame.draw.ellipse(screen, (18,22,26), (x-7, y+6, 14, 5))
    # body
    by = y + int(1 - bob)
    pygame.draw.rect(screen, body, (x-6, by, 12, 8), border_radius=3)
    # legs
    if stepping:
        leg_off = 2 if (walk_phase < 0.5) else -2
        pygame.draw.rect(screen, (18,22,26), (x-4, by+6, 2, 4), border_radius=1)
        pygame.draw.rect(screen, (18,22,26), (x+2, by+6+leg_off, 2, 4), border_radius=1)
    else:
        pygame.draw.rect(screen, (18,22,26), (x-3, by+7, 2, 3), border_radius=1)
        pygame.draw.rect(screen, (18,22,26), (x+2, by+7, 2, 3), border_radius=1)
    # head + ears
    hy = y - 6 - int(bob)
    pygame.draw.rect(screen, skin, (x-7, hy-6, 14, 12), border_radius=3)
    pygame.draw.rect(screen, (20,20,26), (x-7, hy-6, 14, 12), 1, border_radius=3)
    pygame.draw.rect(screen, skin, (x-10, hy-6, 4, 6), border_radius=2)
    pygame.draw.rect(screen, skin, (x+6,  hy-6, 4, 6), border_radius=2)
    # eyes
    blink = ((int((pygame.time.get_ticks()/333) + (dong.id%13)*0.13) % 40) == 0)
    if blink:
        pygame.draw.rect(screen, (20,20,26), (x-3, hy-1, 2, 2))
        pygame.draw.rect(screen, (20,20,26), (x+2, hy-1, 2, 2))
    else:
        pygame.draw.circle(screen, (30,30,36), (x-2, hy-1), 2)
        pygame.draw.circle(screen, (30,30,36), (x+3, hy-1), 2)

def draw_bubble(screen, x,y, text_emoji:str):
    pad=6; w=max(48, 14*len(text_emoji)+pad*2); h=24
    rect = pygame.Rect(int(x-w/2), int(y-34), w, h)
    pygame.draw.rect(screen, (24,28,36), rect, border_radius=8)
    pygame.draw.rect(screen, (70,90,120), rect, 2, border_radius=8)
    pygame.draw.polygon(screen, (24,28,36),
                        [(rect.centerx-6, rect.bottom-2),
                         (rect.centerx+6, rect.bottom-2),
                         (rect.centerx, rect.bottom+6)])
    tfont = pygame.font.SysFont("segoe ui emoji", 18)
    screen.blit(tfont.render(text_emoji, True, (240,240,255)), (rect.x+pad, rect.y+2))

def draw_terminal_block(screen, term):
    x, y = int(term.x), int(term.y)
    pygame.draw.ellipse(screen, (18,22,26), (x-26, y+12, 52, 12))
    pygame.draw.rect(screen, (26,30,36), (x-6, y, 12, 14), border_radius=3)
    kb = pygame.Rect(x-24, y+8, 48, 12)
    pygame.draw.rect(screen, (22,26,30), kb, border_radius=4)
    pygame.draw.rect(screen, (70,90,120), kb, 2, border_radius=4)
    for row_y in (kb.y+3, kb.y+7):
        for k in range(6):
            pygame.draw.rect(screen, (200,210,220), (kb.x+5+k*7, row_y, 5, 2), border_radius=1)

    box = pygame.Rect(x-26, y-28, 52, 36)
    pygame.draw.rect(screen, (28,32,38), box, border_radius=6)
    pygame.draw.rect(screen, (70,90,120), box, 2, border_radius=6)

    screen_rect = pygame.Rect(box.x+5, box.y+6, box.w-10, box.h-14)
    pygame.draw.rect(screen, (18,40,50), screen_rect, border_radius=4)

    for sy in range(screen_rect.y+2, screen_rect.bottom-2, 4):
        pygame.draw.line(screen, (12,26,30), (screen_rect.x+2, sy), (screen_rect.right-2, sy))

    if getattr(term, "flash_until", 0) > pygame.time.get_ticks()/1000.0:
        midy = (screen_rect.y + screen_rect.bottom)//2
        pygame.draw.line(screen, (90, 240, 140), (screen_rect.x+4, midy), (screen_rect.right-4, midy), 2)

    led = (90,220,120) if getattr(term,"occupied_by",None) else (60,70,80)
    pygame.draw.circle(screen, led, (box.right-10, box.y+10), 3)

def draw_toys(screen, toys):
    for it in toys.items:
        x,y = int(it["x"]), int(it["y"])
        if it["kind"] == "ball":
            pygame.draw.circle(screen, TOY_EDGE, (x,y), 10)
            pygame.draw.circle(screen, TOY_BALL_2, (x,y), 9)
            pygame.draw.circle(screen, TOY_BALL_1, (x-2,y-2), 8)
            pygame.draw.arc(screen, TOY_STRIPE, (x-8,y-6,16,12), 0.6, 2.6, 2)
            pygame.draw.circle(screen, (255,255,255), (x-4,y-4), 2)
            if it["carrier"]:
                pygame.draw.line(screen, (210,230,255), (x-8,y+6), (x+8,y+6), 1)
        else:
            r = pygame.Rect(x-10, y-10, 20, 20)
            pygame.draw.rect(screen, TOY_EDGE, r.inflate(4,4), border_radius=4)
            pygame.draw.rect(screen, TOY_BLOCK, r, border_radius=4)

def draw_breeding_curtains(screen, sessions):
    for sess in sessions:
        a,b = sess["a"], sess["b"]
        xa,ya = int(a.x), int(a.y); xb,yb = int(b.x), int(b.y)
        cx, cy = (xa+xb)//2, (ya+yb)//2
        dist = max(16, min(56, int(((xa-xb)**2 + (ya-yb)**2)**0.5)))
        rect = pygame.Rect(cx - dist//2, cy - 12, dist, 24)
        shade = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        shade.fill((8,10,14, 160))
        screen.blit(shade, rect.topleft)
        pygame.draw.rect(screen, (70,90,120), rect, 2, border_radius=6)
        pygame.draw.line(screen, (160,170,180), (rect.left, rect.top), (rect.right, rect.top), 2)
        pygame.draw.circle(screen, (220,200,100), (rect.left, rect.top), 3)
        pygame.draw.circle(screen, (220,200,100), (rect.right, rect.top), 3)

def draw_death_markers(screen, markers, nn_step):
    """Draw small black crosses where Dongs died; persist ~150 NN steps."""
    for m in markers:
        if m["expire"] <= nn_step:  # (kept but not drawn if expired; population prunes too)
            continue
        x, y = int(m["x"]), int(m["y"])
        col = (10,10,12)
        size = 8
        pygame.draw.line(screen, col, (x-size, y), (x+size, y), 2)
        pygame.draw.line(screen, col, (x, y-size), (x, y+size), 2)
