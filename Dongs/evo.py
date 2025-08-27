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

        # slow grass patches
        self.speed = np.ones((self.gw, self.gh), dtype=np.float32)
        for _ in range(90):
            cx, cy = self.rng.randrange(self.gw), self.rng.randrange(self.gh)
            rad = self.rng.randrange(2,5)
            for x in range(max(0,cx-rad), min(self.gw, cx+rad+1)):
                for y in range(max(0,cy-rad), min(self.gh, cy+rad+1)):
                    self.speed[x,y] = min(self.speed[x,y], 0.88)

        # lakes
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
    """Fruit spawns near trees so DONGS learn foraging spots."""
    def __init__(self, terrain: Terrain, rng=None, max_items=80, respawn_sec=5.0, near_tree_radius=60):
        self.rng = rng or random.Random()
        self.terrain = terrain
        self.max_items = max_items
        self.respawn_sec = respawn_sec
        self.near_tree_radius = near_tree_radius
        self.items = []
        self._last = time.time()

    def _spawn_near_tree(self):
        if not self.terrain.trees: return
        tx, ty = self.rng.choice(self.terrain.trees)
        for _ in range(20):
            ang = self.rng.random()*math.tau
            r   = self.rng.uniform(6, self.near_tree_radius)
            x   = clamp(tx + math.cos(ang)*r, 12, self.terrain.w-12)
            y   = clamp(ty + math.sin(ang)*r, 12, self.terrain.h-12)
            if not self.terrain.is_water(x,y):
                t = time.time()
                self.items.append((x, y, t))
                return

    def update(self):
        t = time.time()
        if len(self.items) < self.max_items and (t - self._last) > self.respawn_sec:
            self._spawn_near_tree()
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
IN_SIZE = 10
H_MAX   = 12   # upper bound for growth while resting
OUT_SIZE= 6    # [vx, vy, play, social, pray, rest]

def mlp_forward(w1,b1,w2,b2,x, active_h):
    h = np.tanh(np.dot(w1[:active_h, :], x) + b1[:active_h])
    o = np.tanh(np.dot(w2[:, :active_h], h) + b2)
    return o

def new_brain(rng):
    # IN_SIZE -> H_MAX -> OUT_SIZE
    w1 = rng.normal(0,0.6,(H_MAX,IN_SIZE)).astype(np.float32)
    b1 = rng.normal(0,0.6,(H_MAX,)).astype(np.float32)
    w2 = rng.normal(0,0.6,(OUT_SIZE,H_MAX)).astype(np.float32)
    b2 = rng.normal(0,0.6,(OUT_SIZE,)).astype(np.float32)
    return [w1,b1,w2,b2]

def crossover_mutate(rng, A,B, mut=0.05, sigma=0.06):
    """50/50 mix + mutation for NN weights."""
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
        # core drives
        self.max_intel     = float(rng.uniform(0.8, 1.3))
        self.nn_growth     = float(rng.uniform(0.8, 1.2))
        self.pref_eat      = float(rng.uniform(0.6, 1.4))
        self.pref_play     = float(rng.uniform(0.6, 1.4))
        self.pref_breed    = float(rng.uniform(0.6, 1.4))
        self.explore_bias  = float(rng.uniform(0.9, 1.4))
        self.move_scalar   = float(rng.uniform(0.90, 1.15))
        # social/rest/learning
        self.sociality     = float(rng.uniform(0.7, 1.3))
        self.rest_need     = float(rng.uniform(0.8, 1.2))
        self.brain_plastic = float(rng.uniform(0.8, 1.3))
        # will to live / survival drive
        self.survival_drive= float(rng.uniform(0.7, 1.3))

    def mix_with(self, other, rng, mut_sigma=0.03):
        """50/50 per-gene with mild mutation."""
        g = Genes(rng)
        for k,v in vars(self).items():
            if k.startswith("_"): continue
            ov = getattr(other, k)
            pick = v if (rng.random()<0.5) else ov
            setattr(g, k, max(0.1, float(pick + rng.gauss(0, mut_sigma))))
        return g

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

        # network growth
        self.active_h = int(clamp(rng.randint(6, 8), 2, H_MAX))
        self.last_rest_learn = 0.0

        # behavior state
        self.bubble=None; self.last_bubble_time=0.0
        self.waiting_for_reply=False
        self.docked_terminal=False
        self.is_praying=False
        self.pending_prayer=None
        self.outbox_msg=None
        self.chat_started_at=0.0
        self.breeding_until = 0.0
        self.last_breed_time = 0.0
        self.boredom = 0.0
        self.rest_until = 0.0

        # terminal use limits
        self.terminal_dock_since = 0.0
        self.terminal_cooldown_until = 0.0

        # social memory & boosts
        self.affinity = {}              # other_id -> score
        self.last_social_int = 0.0
        self.recent_social_reward = 0.0 # decays during rest; boosts plasticity
        self.last_gift_time = 0.0

        # exploration waypoint
        self.tx = self.rng.uniform(60, 1220)
        self.ty = self.rng.uniform(60, 660)
        self.change_target_at = time.time() + self.rng.uniform(10, 22)

    @property
    def display_id(self) -> str:
        return f"DONG{self.id+1:04d}"

    def compose_reply(self, user_text: str) -> str:
        """Create a short 'donglish' utterance the LLM will translate."""
        syll = ["do", "ng", "ka", "la", "mi", "bo", "ti", "ra", "su", "po"]
        w = self.rng.randint(3, 5)
        base = "".join(self.rng.choice(syll) for _ in range(w))
        tags = []
        if "food" in user_text.lower(): tags.append("nom")
        if "play" in user_text.lower(): tags.append("pla")
        if "friend" in user_text.lower(): tags.append("fri")
        if "learn" in user_text.lower(): tags.append("lea")
        if not tags: tags.append(self.rng.choice(["fri","pla","nom","lea"]))
        return f"{base}-{tags[0]}~"

    def _maybe_new_target(self):
        if time.time() >= self.change_target_at or (abs(self.x-self.tx)+abs(self.y-self.ty) < 40):
            self.tx = self.rng.uniform(60, 1220)
            self.ty = self.rng.uniform(60, 660)
            self.change_target_at = time.time() + self.rng.uniform(10, 22)

    def desire_to_breed(self):
        drive = 7.6 * self.genes.pref_breed
        cd_ok = (time.time() - self.last_breed_time) > 14.0
        return (self.energy > drive) and (self.age > 12.0) and cd_ok

    def _rest_and_learn(self, dt):
        """Offline learning during rest; can 'grow' hidden neurons."""
        now = time.time()
        if now - self.last_rest_learn > 0.4:
            self.last_rest_learn = now
            w1,b1,w2,b2 = self.brain
            # Social interactions increase plasticity a little (decays here)
            social_boost = 1.0 + min(1.0, self.recent_social_reward*0.5)
            noise = 0.005 * self.genes.brain_plastic * social_boost
            w1[:self.active_h,:] += np.random.normal(0, noise, w1[:self.active_h,:].shape).astype(np.float32)
            b1[:self.active_h]   += np.random.normal(0, noise, b1[:self.active_h].shape).astype(np.float32)
            w2[:, :self.active_h]+= np.random.normal(0, noise, w2[:, :self.active_h].shape).astype(np.float32)
            if self.active_h < H_MAX and self.rng.random() < 0.10 * self.genes.nn_growth * social_boost:
                self.active_h += 1
            # decay recent reward
            self.recent_social_reward *= 0.85
        # Rest recovers energy, reduces boredom
        self.energy = min(10.0, self.energy + 0.01 * dt * 60)
        self.boredom = max(0.0, self.boredom - 0.02 * dt * 60)

    def _read_peer_bubbles(self, neighbors, terminal):
        """Read nearby bubbles to adjust affinity/targets/learning."""
        for nb in neighbors:
            if not nb.bubble: continue
            if (nb.x-self.x)**2 + (nb.y-self.y)**2 > 44**2: 
                continue
            emo = nb.bubble
            # Friendship / chat bubbles
            if emo in ("🗨️","🤝","🎁","💞"):
                self.affinity[nb.id] = self.affinity.get(nb.id,0.0) + 0.05
                self.recent_social_reward += 0.1
                # sometimes drift toward friendly peer
                if random.random() < 0.2:
                    self.tx, self.ty = nb.x, nb.y
            # Rest bubble
            if emo == "💤":
                # mirrors restful mood: reduce boredom slightly
                self.boredom = max(0.0, self.boredom - 0.1)
            # Terminal/prayer bubbles nudge curiosity toward terminal
            if emo in ("🖥️","🙏"):
                if random.random() < 0.15:
                    self.tx, self.ty = terminal.x, terminal.y

    def step(self, dt, terrain: Terrain, food, neighbors, terminal, toys):
        now = time.time()

        # frozen states still AGE at half rate while motionless at terminal/prayer/breeding
        if self.waiting_for_reply or self.is_praying or now < self.breeding_until:
            self.age += dt * 0.5
            self.vx=self.vy=0.0
            return

        # Will-to-live modulates rest threshold (more survival drive -> rest a bit sooner)
        tired_threshold = 3.2 * self.genes.rest_need * (0.9 + 0.2*self.genes.survival_drive)
        tired = self.energy < tired_threshold
        bored = self.boredom > 8.0
        if (tired or bored) and now < self.rest_until:
            self._rest_and_learn(dt)
            self.age += dt  # normal age while resting
            self.vx=self.vy=0.0; return
        if (tired or bored) and now >= self.rest_until:
            self.rest_until = now + self.rng.uniform(1.6, 3.4)
            self.bubble="💤"; self.last_bubble_time=now
            self.age += dt
            self.vx=self.vy=0.0; return

        self._maybe_new_target()
        self._read_peer_bubbles(neighbors, terminal)

        # nearest food
        nf=None; nfd=1e9
        for (fx,fy,_) in food.items:
            d=(fx-self.x)**2+(fy-self.y)**2
            if d<nfd: nfd, nf = d, (fx,fy)
        if nf is None: nf=(self.x,self.y)
        fx,fy=nf

        # nearest peer
        peer=None; pd2=1e9
        for p in neighbors:
            d=(p.x-self.x)**2+(p.y-self.y)**2
            if d<pd2: pd2=d; peer=p
        px,py = (peer.x,peer.y) if peer else (self.x,self.y)

        # inputs (10) with survival-aware food weighting
        Rf = 220.0; Rp = 220.0; Rt = 260.0
        hunger_amp = 1.0 + self.genes.survival_drive * clamp((6.0 - self.energy)/6.0, 0.0, 1.0)
        food_pref = self.genes.pref_eat * hunger_amp
        x = np.array([
            (fx-self.x)/Rf * food_pref,
            (fy-self.y)/Rf * food_pref,
            len(neighbors)/6.0,
            (px-self.x)/Rp,
            (py-self.y)/Rp,
            (terminal.x-self.x)/Rt,
            (terminal.y-self.y)/Rt,
            1.0 if (getattr(terminal,"recent_msg_until",0) > now and
                    (self.x-terminal.x)**2 + (self.y-terminal.y)**2 < (terminal.radius+80)**2) else 0.0,
            self.energy/10.0,
            self.boredom/10.0
        ], dtype=np.float32)

        # strong edge repel + centre pull + waypoint attraction
        margin = 80.0
        repx = repy = 0.0
        if self.x < margin:           repx = (margin - self.x) * 0.020
        elif self.x > terrain.w-margin:repx = -(self.x - (terrain.w-margin)) * 0.020
        if self.y < margin:           repy = (margin - self.y) * 0.020
        elif self.y > terrain.h-margin:repy = -(self.y - (terrain.h-margin)) * 0.020
        avoid_scale = 0.8 + 0.4*self.genes.survival_drive
        repx *= avoid_scale; repy *= avoid_scale

        cx_pull = ((terrain.w*0.5 - self.x) / 420.0) * 0.9
        cy_pull = ((terrain.h*0.5 - self.y) / 420.0) * 0.9
        wx = (self.tx - self.x) / 260.0 * self.genes.explore_bias
        wy = (self.ty - self.y) / 260.0 * self.genes.explore_bias

        # forward pass
        w1,b1,w2,b2 = self.brain
        out = mlp_forward(w1,b1,w2,b2,x, self.active_h)
        ax = float(out[0] + repx + cx_pull + wx)
        ay = float(out[1] + repy + cy_pull + wy)

        play_int   = float(out[2]) * self.genes.pref_play
        social_int = float(out[3]) * self.genes.sociality
        pray_int   = float(out[4]) * self.genes.max_intel
        rest_int   = float(out[5]) * self.genes.rest_need

        # survival pressure slightly suppresses prayer if energy is low
        pray_int *= (0.8 + 0.2*(self.energy/10.0))
        pray_int *= (1.0 - 0.2*self.genes.survival_drive * clamp((6.0 - self.energy)/6.0, 0.0, 1.0))
        self.last_social_int = social_int

        # movement
        spd = 54.0 * self.genes.move_scalar * terrain.speed_at(self.x, self.y)
        self.vx = ax * spd * dt; self.vy = ay * spd * dt
        nx, ny = self.x + self.vx, self.y + self.vy
        if not terrain.is_water(nx, ny):
            self.x, self.y = nx, ny

        # social interaction nearby -> builds affinity and boosts later learning
        if neighbors and social_int > 0.55:
            peer = peer or self.rng.choice(neighbors)
            if (peer.x-self.x)**2 + (peer.y-self.y)**2 < 22**2:
                self.bubble="🗨️"; self.last_bubble_time=now
                self.fitness += 0.004 * self.genes.nn_growth
                self.boredom = max(0.0, self.boredom - 0.6)
                delta = 0.02 * max(0.3, social_int)
                self.affinity[peer.id] = self.affinity.get(peer.id, 0.0) + delta
                peer.affinity[self.id] = peer.affinity.get(self.id, 0.0) + delta
                self.recent_social_reward += 0.2
                peer.recent_social_reward += 0.2
                if self.affinity[peer.id] > 0.8 and self.rng.random() < 0.1:
                    self.bubble="🤝"; self.last_bubble_time=now

        # toy play / carry + gifting to peers
        nearest_peer = peer
        for it in getattr(toys, "items", []):
            dx,dy = it["x"]-self.x, it["y"]-self.y
            d2 = dx*dx + dy*dy
            if it["kind"] == "ball":
                # gifting: pass a carried ball to a nearby peer
                if it["carrier"] is self and nearest_peer and ((nearest_peer.x-self.x)**2 + (nearest_peer.y-self.y)**2) < 18**2:
                    if (now - self.last_gift_time) > 4.0 and self.rng.random() < 0.06:
                        it["carrier"] = nearest_peer
                        self.last_gift_time = now
                        self.affinity[nearest_peer.id] = self.affinity.get(nearest_peer.id, 0.0) + 0.35
                        nearest_peer.affinity[self.id] = nearest_peer.affinity.get(self.id, 0.0) + 0.35
                        self.bubble="🎁"; self.last_bubble_time=now
                        nearest_peer.bubble="🎁"; nearest_peer.last_bubble_time=now
                        self.recent_social_reward += 0.3; nearest_peer.recent_social_reward += 0.3

                if it["carrier"] is self:
                    if self.energy < 5.0 and self.rng.random() < 0.02:
                        it["carrier"] = None
                    elif self.rng.random() < 0.005:
                        it["carrier"] = None
                    else:
                        self.fitness += 0.002 * self.genes.nn_growth
                        self.boredom = max(0.0, self.boredom - 0.1)
                elif d2 < 15**2:
                    if it["carrier"] is None and self.energy > 6.0 and play_int > 0.55 and self.rng.random() < 0.12*self.genes.pref_play:
                        it["carrier"] = self
                        self.bubble = "🟦"; self.last_bubble_time = now
                        self.fitness += 0.01 * self.genes.nn_growth
                        self.boredom = max(0.0, self.boredom - 1.0)
                    else:
                        it["vx"] += dx*0.04; it["vy"] += dy*0.04
                        self.fitness += 0.001 * self.genes.nn_growth
            else:
                if d2 < 16**2:
                    self.energy = min(10.0, self.energy+0.02)
                    self.fitness += 0.0008 * self.genes.nn_growth
                    self.boredom = max(0.0, self.boredom - 0.2)

        # eat & metabolism
        self.energy -= 0.0024 + 0.0012*abs(ax) + 0.0012*abs(ay)
        if nfd < 16**2:
            for i,(fx2,fy2,_) in enumerate(list(food.items)):
                if (fx2-self.x)**2 + (fy2-self.y)**2 < 16**2:
                    self.energy = min(10.0, self.energy+2.2); food.items.pop(i)
                    self.boredom = max(0.0, self.boredom - 1.0)
                    break

        self.age += dt
        self.fitness = min(self.fitness + 0.001*self.genes.max_intel, 9999.0)
        self.boredom = min(10.0, self.boredom + 0.002*60*dt)

        # ---- terminal decisions (one-at-a-time enforced by terminal.occupied_by) ----
        near_term = (self.x-terminal.x)**2 + (self.y-terminal.y)**2 < (terminal.radius+14)**2
        if near_term and terminal.occupied_by is None and self.energy > 4.0 and time.time() >= self.terminal_cooldown_until:
            # Curiosity/user-chat vs prayer choice
            chat_int = social_int * self.genes.max_intel * 0.8 + 0.2*self.genes.explore_bias
            if chat_int > pray_int and self.rng.random() < 0.28:
                terminal.occupied_by = self
                self.waiting_for_reply=True
                self.docked_terminal=True
                self.chat_started_at=time.time()
                self.terminal_dock_since = self.chat_started_at
                self.bubble="🖥️"; self.last_bubble_time=now
                starters = [
                    "hel-do fri~", "mi pla!~", "ra su po?", "ka bo ti~",
                    "dong-dong lea?", "we wander~", "find nom?"
                ]
                self.outbox_msg = f"{self.display_id}: {self.rng.choice(starters)}"
                return

        # dock for USER chat if the user just typed something
        if (not self.waiting_for_reply and not self.docked_terminal and
            getattr(terminal,"recent_msg_until",0) > now and
            near_term and terminal.occupied_by is None and social_int > 0.6 and self.energy > 4.0
            and time.time() >= self.terminal_cooldown_until):
            terminal.occupied_by = self
            self.waiting_for_reply=True; self.docked_terminal=True
            self.chat_started_at=time.time()
            self.terminal_dock_since = self.chat_started_at
            self.bubble="🖥️"; self.last_bubble_time=now

        # pray to LLM "God" (only near terminal and only if terminal free)
        if (not self.is_praying and not self.docked_terminal and
            near_term and terminal.occupied_by is None and pray_int > 0.65 and self.energy > 4.5
            and time.time() >= self.terminal_cooldown_until):
            terminal.occupied_by = self
            self.is_praying = True
            self.pending_prayer = self._make_prayer()
            self.terminal_dock_since = time.time()
            self.bubble="🙏"; self.last_bubble_time=now

        # idle emotes
        if self.rng.random() < 0.002:
            self.bubble = "🙂" if self.energy>6 else "😕"; self.last_bubble_time=now

    def _make_prayer(self):
        topics = [
            "guide my steps", "where is food", "teach friendship", "we seek wisdom",
            "how to grow brain", "help us play better", "how to rest well",
            "why are we small", "how to learn fast"
        ]
        return f"{self.display_id} prays: {self.rng.choice(topics)}"

# ---------- Population ----------
class Population:
    def __init__(self, terrain: Terrain, rng=None, max_pop=50):
        self.rng = rng or random.Random()
        self.terrain = terrain
        self.max_pop = max_pop
        self.bots = []
        self.breeding_sessions = []
        self.bred_pairs = set()  # {(min_id,max_id)}
        self.death_marks = []    # list of dicts: {"x":..,"y":..,"expire": step}

    def reset(self, max_pop=None):
        self.max_pop = max_pop or self.max_pop
        self.bots.clear(); self.breeding_sessions.clear()
        self.bred_pairs.clear()
        self.death_marks.clear()
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

    def step(self, dt, terrain: Terrain, food: Food, terminal, toys, nn_step: int):
        now = time.time()

        # complete breeding sessions -> produce 1-2 kids with 50/50 genes and NN
        done=[]
        for sess in list(self.breeding_sessions):
            a,b = sess["a"], sess["b"]
            a.breeding_until = sess["until"]; b.breeding_until = sess["until"]
            if now >= sess["until"]:
                for p in (a,b):
                    p.energy = max(0.5, p.energy * np.random.uniform(0.6, 0.7))  # cost
                    p.last_breed_time = time.time()
                n_children = sess["n_children"]
                for _ in range(n_children):
                    nx = (a.x+b.x)/2 + self.rng.uniform(-6,6)
                    ny = (a.y+b.y)/2 + self.rng.uniform(-6,6)
                    if not terrain.is_water(nx,ny) and len(self.bots) < self.max_pop:
                        child_brain = crossover_mutate(np.random, a.brain, b.brain, mut=0.06, sigma=0.08)
                        child_genes = a.genes.mix_with(b.genes, self.rng, mut_sigma=0.02)
                        kid = Bot(self.rng, nx, ny, child_brain, child_genes)
                        kid.energy = 6.0; kid.fitness += 0.02
                        self.bots.append(kid)
                done.append(sess)
        for d in done: 
            if d in self.breeding_sessions:
                self.breeding_sessions.remove(d)

        # step individuals
        for i,b in enumerate(self.bots):
            neighbors = [o for o in self.bots if o is not b and (o.x-b.x)**2+(o.y-b.y)**2 < 40**2]
            b.step(dt, terrain, food, neighbors, terminal, toys)

        # restricted breeding: stricter, affinity-weighted, pair lockout
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
                    pair_key = (min(a.id, c.id), max(a.id, c.id))
                    if pair_key in self.bred_pairs:  # already bred once together
                        continue
                    # nearby?
                    if (a.x-c.x)**2+(a.y-c.y)**2 >= 36**2: 
                        continue
                    # social-affinity + capabilities gate (stricter)
                    aff = 0.5*(a.affinity.get(c.id,0.0) + c.affinity.get(a.id,0.0))
                    soc = 0.5*(a.last_social_int + c.last_social_int)
                    skill = 0.02*(a.active_h + c.active_h) + 0.02*(a.fitness + c.fitness)
                    will = 0.1*(a.genes.survival_drive + c.genes.survival_drive)
                    score = 0.65*aff + 0.2*soc + skill + will
                    if score < 1.05:
                        continue
                    if len(self.breeding_sessions) >= 4:
                        continue
                    # pair accepted
                    n_children = self.rng.randint(1,2)
                    dur = 2.4; until = time.time()+dur
                    a.bubble="💞"; a.last_bubble_time=time.time()
                    c.bubble="💞"; c.last_bubble_time=time.time()
                    a.breeding_until=until; c.breeding_until=until
                    self.breeding_sessions.append({"a":a,"b":c,"until":until,"n_children":n_children})
                    self.bred_pairs.add(pair_key)
                    used.add(i); used.add(j)
                    break

        # death markers and cleanup
        dead = [b for b in self.bots if (b.energy<=0.0 or b.age >= b.max_age)]
        for b in dead:
            self.death_marks.append({"x": b.x, "y": b.y, "expire": nn_step + 150})
        self.bots[:] = [b for b in self.bots if b not in dead]
        # release terminal if occupant died
        if terminal.occupied_by and terminal.occupied_by not in self.bots:
            terminal.occupied_by = None
        # prune old crosses
        self.death_marks[:] = [m for m in self.death_marks if m["expire"] > nn_step]
