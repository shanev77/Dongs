import math, random, time
import numpy as np

def clamp(v, lo, hi): return lo if v<lo else (hi if v>hi else v)

# ---------- Terrain ----------
class Terrain:
    def __init__(self, W, H, cell=20, rng=None):
        self.rng = rng or random.Random()
        self.w, self.h = W, H
        self.cell = cell
        self.gw, self.gh = W//cell, H//cell

        self.speed = np.ones((self.gw, self.gh), dtype=np.float32)
        for _ in range(90):
            cx, cy = self.rng.randrange(self.gw), self.rng.randrange(self.gh)
            rad = self.rng.randrange(2,5)
            for x in range(max(0,cx-rad), min(self.gw, cx+rad+1)):
                for y in range(max(0,cy-rad), min(self.gh, cy+rad+1)):
                    self.speed[x,y] = min(self.speed[x,y], 0.88)

        self.water = []
        for _ in range(8):
            cx = self.rng.randrange(self.gw//6, self.gw-self.gw//6)
            cy = self.rng.randrange(self.gh//7, self.gh-self.gh//7)
            r  = self.rng.randrange(2,4)
            self.water.append((cx, cy, r))

        self.speckles = [(self.rng.randrange(0,W), self.rng.randrange(0,H)) for _ in range(260)]
        self.trees = [(self.rng.randrange(30,W-30), self.rng.randrange(30,H-30)) for _ in range(28)]
        self.rocks = [(self.rng.randrange(20,W-20), self.rng.randrange(20,H-20)) for _ in range(34)]

    def is_water(self, px, py):
        for cx,cy,r in self.water:
            x = cx*self.cell + self.cell//2
            y = cy*self.cell + self.cell//2
            if (px-x)**2 + (py-y)**2 <= (r*self.cell//2)**2:
                return True
        return False

    def speed_at(self, px, py):
        gx = clamp(int(px//self.cell), 0, self.gw-1)
        gy = clamp(int(py//self.cell), 0, self.gh-1)
        return float(self.speed[gx,gy])

# ---------- Food ----------
class Food:
    def __init__(self, rng=None, max_items=80, respawn_sec=5.0):
        self.rng = rng or random.Random()
        self.max_items = max_items
        self.respawn_sec = respawn_sec
        self.items = []
        self._last = time.time()
    def update(self):
        t = time.time()
        if len(self.items) < self.max_items and (t - self._last) > self.respawn_sec:
            self.items.append((self.rng.randrange(20, 1260), self.rng.randrange(20, 700), t))
            self._last = t

# ---------- Toys ----------
class Toys:
    def __init__(self, rng=None, count=8):
        self.rng = rng or random.Random()
        self.items = []
        for _ in range(count):
            kind = "ball" if self.rng.random()<0.7 else "block"
            self.items.append({"kind":kind, "x": self.rng.randrange(40,1240), "y": self.rng.randrange(40,680),
                               "vx":0.0, "vy":0.0, "carrier": None})
    def update(self, dt):
        for it in self.items:
            c = it["carrier"]
            if c is not None:
                offx = 10 * math.cos(time.time()*4.0 + (c.id%7))
                offy = -12 + 2 * math.sin(time.time()*6.0 + (c.id%5))
                it["x"] = c.x + offx
                it["y"] = c.y + offy
                it["vx"] = it["vy"] = 0.0
            else:
                if it["kind"]=="ball":
                    it["x"] += it["vx"]*dt; it["y"] += it["vy"]*dt
                    it["vx"] *= 0.96; it["vy"] *= 0.96
                it["x"] = clamp(it["x"], 20, 1260)
                it["y"] = clamp(it["y"], 20, 700)

# ---------- Brain ----------
def mlp_forward(w1,b1,w2,b2,x):
    h = np.tanh(np.dot(w1, x) + b1)
    o = np.tanh(np.dot(w2, h) + b2)
    return o

def new_brain(rng):
    w1 = rng.normal(0,0.6,(8,4)).astype(np.float32)
    b1 = rng.normal(0,0.6,(8,)).astype(np.float32)
    w2 = rng.normal(0,0.6,(2,8)).astype(np.float32)
    b2 = rng.normal(0,0.6,(2,)).astype(np.float32)
    return [w1,b1,w2,b2]

def crossover_mutate(rng, A,B, mut=0.05, sigma=0.06):
    out=[]
    for a,b in zip(A,B):
        mask = np.random.rand(*a.shape) < 0.5
        child = np.where(mask, a, b).astype(np.float32)
        m = (np.random.rand(*child.shape) < mut).astype(np.float32)
        child += m * np.random.normal(0, sigma, child.shape).astype(np.float32)
        out.append(child)
    return out

# ---------- Genes ----------
class Genes:
    def __init__(self, rng):
        self.max_intel   = float(rng.uniform(0.8, 1.3))
        self.nn_growth   = float(rng.uniform(0.8, 1.2))
        self.pref_eat    = float(rng.uniform(0.6, 1.4))
        self.pref_play   = float(rng.uniform(0.6, 1.4))
        self.pref_breed  = float(rng.uniform(0.6, 1.4))
        self.explore_bias= float(rng.uniform(0.9, 1.4))
        self.move_scalar = float(rng.uniform(0.90, 1.15))

# ---------- Bot ----------
class Bot:
    _idseq = 0
    def __init__(self, rng, x,y, brain, genes: Genes):
        self.id = Bot._idseq; Bot._idseq += 1
        self.rng = rng
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.size = 9
        self.energy = 8.0
        self.fitness = 0.0
        self.age = 0.0
        self.max_age = rng.uniform(100,150) * genes.max_intel
        self.brain = brain
        self.genes = genes

        self.wander = np.array([self.rng.uniform(-1,1), self.rng.uniform(-1,1)], dtype=np.float32)

        self.bubble=None; self.last_bubble_time=0.0
        self.waiting_for_reply=False
        self.docked_terminal=False
        self.breeding_until = 0.0
        self.last_breed_time = 0.0

        # persistent exploration waypoint
        self.tx = self.rng.uniform(40, 1240)
        self.ty = self.rng.uniform(40, 680)
        self.change_target_at = time.time() + self.rng.uniform(8, 18)

    @property
    def display_id(self) -> str:
        return f"DONG{self.id+1:04d}"

    def _maybe_new_target(self):
        if time.time() >= self.change_target_at or (abs(self.x-self.tx)+abs(self.y-self.ty) < 40):
            self.tx = self.rng.uniform(60, 1220)
            self.ty = self.rng.uniform(60, 660)
            self.change_target_at = time.time() + self.rng.uniform(10, 22)

    def desire_to_breed(self):
        drive = 7.6 * self.genes.pref_breed
        return (self.energy > drive) and (time.time() - self.last_breed_time > 12.0) and (self.age > 12.0)

    def step(self, dt, terrain: Terrain, food: Food, neighbors, terminal, toys):
        now = time.time()
        if self.waiting_for_reply or now < self.breeding_until:
            self.vx=self.vy=0.0; return

        self._maybe_new_target()

        # nearest food
        nf=None; nfd=9999
        for (fx,fy,_) in food.items:
            d=(fx-self.x)**2+(fy-self.y)**2
            if d<nfd: nfd, nf = d, (fx,fy)
        if nf is None: nf=(self.x,self.y)
        fx,fy=nf

        # strong edge repel + centre pull + waypoint attraction + wander
        margin = 80.0
        repx = 0.0; repy = 0.0
        if self.x < margin:           repx = (margin - self.x) * 0.020
        elif self.x > terrain.w-margin:repx = -(self.x - (terrain.w-margin)) * 0.020
        if self.y < margin:           repy = (margin - self.y) * 0.020
        elif self.y > terrain.h-margin:repy = -(self.y - (terrain.h-margin)) * 0.020

        cx_pull = ((terrain.w*0.5 - self.x) / 420.0) * 0.9
        cy_pull = ((terrain.h*0.5 - self.y) / 420.0) * 0.9

        # waypoint (exploration)
        wx = (self.tx - self.x) / 260.0 * self.genes.explore_bias
        wy = (self.ty - self.y) / 260.0 * self.genes.explore_bias

        # gentle wander noise
        self.wander += np.array([np.random.normal(0,0.08), np.random.normal(0,0.08)], dtype=np.float32)
        self.wander *= 0.92

        x = np.array([
            (fx-self.x)/220.0 * self.genes.pref_eat + wx + repx + cx_pull,
            (fy-self.y)/220.0 * self.genes.pref_eat + wy + repy + cy_pull,
            self.energy/10.0,
            len(neighbors)/6.0
        ], dtype=np.float32)

        w1,b1,w2,b2 = self.brain
        out = mlp_forward(w1,b1,w2,b2,x)
        ax = float(out[0] + 0.34*self.wander[0])
        ay = float(out[1] + 0.34*self.wander[1])

        spd = 54.0 * self.genes.move_scalar * terrain.speed_at(self.x, self.y)
        self.vx = ax * spd * dt; self.vy = ay * spd * dt

        nx, ny = self.x + self.vx, self.y + self.vy
        if not terrain.is_water(nx, ny):
            self.x, self.y = nx, ny

        # play with toys
        for it in toys.items:
            dx,dy = it["x"]-self.x, it["y"]-self.y
            d2 = dx*dx + dy*dy
            if it["kind"] == "ball":
                if it["carrier"] is self:
                    if self.energy < 5.0 and self.rng.random() < 0.02:
                        it["carrier"] = None
                    elif self.rng.random() < 0.005:
                        it["carrier"] = None
                    else:
                        self.fitness += 0.002 * self.genes.nn_growth
                elif d2 < 15**2:
                    if it["carrier"] is None and self.energy > 6.0 and self.rng.random() < 0.12*self.genes.pref_play:
                        it["carrier"] = self
                        self.bubble = "🟦"; self.last_bubble_time = now
                        self.fitness += 0.01 * self.genes.nn_growth
                    else:
                        it["vx"] += dx*0.04; it["vy"] += dy*0.04
                        self.fitness += 0.001 * self.genes.nn_growth
            else:
                if d2 < 16**2:
                    self.energy = min(10.0, self.energy+0.02)
                    self.fitness += 0.0008 * self.genes.nn_growth

        # eat
        self.energy -= 0.002 + 0.001*abs(ax) + 0.001*abs(ay)
        if nfd < 16**2:
            for i,(fx2,fy2,_) in enumerate(list(food.items)):
                if (fx2-self.x)**2 + (fy2-self.y)**2 < 16**2:
                    self.energy = min(10.0, self.energy+2.2); food.items.pop(i); break

        self.age += dt; self.fitness = min(self.fitness + 0.001*self.genes.max_intel, 9999.0)
        if self.rng.random() < 0.002:
            self.bubble = "🙂" if self.energy>6 else "😕"; self.last_bubble_time=now

        # self-initiated terminal chat (rare)
        if not self.waiting_for_reply and not self.docked_terminal:
            if (self.rng.random() < 0.0007) or (self.energy<4.0 and self.rng.random()<0.0011):
                if (self.x-terminal.x)**2 + (self.y-terminal.y)**2 < (terminal.radius+12)**2:
                    if terminal.occupied_by is None:
                        terminal.occupied_by = self
                        self.waiting_for_reply=True; self.docked_terminal=True
                        self.bubble="💬"; self.last_bubble_time=now

# ---------- Population ----------
class Population:
    def __init__(self, terrain: Terrain, rng=None, max_pop=50):
        self.rng = rng or random.Random()
        self.terrain = terrain
        self.max_pop = max_pop
        self.bots = []
        self.breeding_sessions = []

    def reset(self, max_pop=None):
        self.max_pop = max_pop or self.max_pop
        self.bots.clear(); self.breeding_sessions.clear()
        Bot._idseq = 0

    def spawn(self, n=10):
        for _ in range(n):
            brain = new_brain(np.random)
            genes = Genes(self.rng)
            for attempt in range(40):
                x = self.rng.randrange(30, self.terrain.w-30)
                y = self.rng.randrange(30, self.terrain.h-30)
                if not self.terrain.is_water(x,y):
                    break
            self.bots.append(Bot(self.rng, x,y, brain, genes))

    def step(self, dt, terrain: Terrain, food: Food, terminal, toys):
        now = time.time()

        done=[]
        for sess in self.breeding_sessions:
            a,b = sess["a"], sess["b"]
            a.breeding_until = sess["until"]; b.breeding_until = sess["until"]
            if now >= sess["until"]:
                for p in (a,b):
                    p.energy = max(0.5, p.energy * np.random.uniform(0.6, 0.7))
                    p.last_breed_time = time.time()
                nx = (a.x+b.x)/2 + self.rng.uniform(-6,6)
                ny = (a.y+b.y)/2 + self.rng.uniform(-6,6)
                if not terrain.is_water(nx,ny) and len(self.bots) < self.max_pop:
                    child_brain = sess["child_brain"]
                    child_genes = Genes(self.rng)
                    kid = Bot(self.rng, nx, ny, child_brain, child_genes)
                    kid.energy = 6.0; kid.fitness += 0.02
                    self.bots.append(kid)
                done.append(sess)
        for d in done: self.breeding_sessions.remove(d)

        for i,b in enumerate(self.bots):
            neighbors = [o for o in self.bots if o is not b and (o.x-b.x)**2+(o.y-b.y)**2 < 40**2]
            b.step(dt, terrain, food, neighbors, terminal, toys)

        if len(self.bots) < self.max_pop:
            order = list(range(len(self.bots))); self.rng.shuffle(order)
            used=set()
            for i in order:
                a=self.bots[i]
                if i in used or not a.desire_to_breed(): continue
                for j in order:
                    if j==i or j in used: continue
                    c=self.bots[j]
                    if not c.desire_to_breed(): continue
                    if (a.x-c.x)**2+(a.y-c.y)**2 < 36**2:
                        child = crossover_mutate(np.random, a.brain, c.brain, mut=0.06, sigma=0.08)
                        dur = 2.4; until = time.time()+dur
                        a.bubble="💞"; a.last_bubble_time=time.time()
                        c.bubble="💞"; c.last_bubble_time=time.time()
                        a.breeding_until=until; c.breeding_until=until
                        self.breeding_sessions.append({"a":a,"b":c,"until":until,"child_brain":child})
                        used.add(i); used.add(j)
                        if len(self.breeding_sessions)>6: break
                if len(self.breeding_sessions)>6: break

        self.bots[:] = [b for b in self.bots if (b.energy>0.0 and b.age < b.max_age)]

        for it in toys.items:
            if it["carrier"] and it["carrier"] not in self.bots:
                it["carrier"] = None

        if terminal.occupied_by and terminal.occupied_by not in self.bots:
            terminal.occupied_by = None
