import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import GenerationConfig
from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper

# export PYTHONPATH=YOUR_M1_PATH

messages = [[
    {
        "role": "user",
        "content": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$",
    },
],  [
    {
        "role": "user",
        "content": "On $\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\overline{AB}$ with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side $\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$.",
    },
],
[
    {
        "role": "user",
        "content": "The 9 members of a baseball team went to an ice cream parlor after their game. Each player had a singlescoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let $N$ be the number of different assignments of flavors to players that meet these conditions. Find the remainder when $N$ is divided by 1000.",
    },
],
[
    {
        "role": "user",
        "content": "There are $8!=40320$ eight-digit positive integers that use each of the digits $1,2,3,4,5,6,7,8$ exactly once. Let $N$ be the number of these integers that are divisible by 22. Find the difference between $N$ and 2025.$",
    },
],
[
    {
        "role": "user",
        "content": "The parabola with equation $y=x^{2}-4$ is rotated $60^{\circ}$ counterclockwise around the origin. The unique point in the fourth quadrant where the original parabola and its image intersect has $y$-coordinate $\frac{a-\sqrt{b}}{c}$, where $a$, $b$, and $c$ are positive integers, and $a$ and $c$ are relatively prime. Find $a+b+c$.",
    },
],
[
    {
        "role": "user",
        "content": "The 27 cells of a $3\times9$ grid are filled in using the numbers 1 through 9 so that each row contains 9 different numbers, and each of the three $3\times3$ blocks heavily outlined in the example below contains 9 different numbers, as in the first three rows of a Sudoku puzzle. | 4 | 2 | 8 | 9 | 6 | 3 | 1 | 7 | 5 | | 3 | 7 | 9 | 5 | 2 | 1 | 6 | 8 | 4 | | 5 | 6 | 1 | 8 | 4 | 7 | 9 | 2 | 3 | The number of different ways to fill such a grid can be written as $p^a\cdot q^b\cdot r^c\cdot s^d$, where $p,q,r,$ and $s$ are distinct prime numbers and $a,b,c,$ and $d$ are positive integers. Find $p\cdot a+q\cdot b+r\cdot c+s\cdot d$."
    },
],
[
    {
        "role": "user",
        "content": "A piecewise linear periodic function is defined by $f(x)=\begin{cases}x&\text{if }x\in[-1,1)\\2-x&\text{if }x\in[1,3)\end{cases}$ and $f(x+4)=f(x)$ for all real numbers $x$. The graph of $f(x)$ has the sawtooth pattern. The parabola $x=34y^2$ intersects the graph of $f(x)$ at finitely many points. The sum of the $y$-coordinates of these intersection points can be expressed in the form $\frac{a+b\sqrt{c}}{d}$, where $a,b,c,$ and $d$ are positive integers, $a,b,$ and $d$ have greatest common divisor equal to 1, and $c$ is not divisible by the square of any prime. Find $a+b+c+d$."    },
],
[
    {
        "role": "user",
        "content": "The set of points in 3-dimensional coordinate space that lie in the plane $x+y+z=75$ whose coordinates satisfy the inequalities $x-yz<y-zx<z-xy$ forms three disjoint convex regions. Exactly one of those regions has finite area. The area of this finite region can be expressed in the form $a\sqrt{b}$, where $a$ and $b$ are positive integers and $b$ is not divisible by the square of any prime. Find $a+b$.",
    },
],
[
    {
        "role": "user",
        "content": "Alex divides a disk into four quadrants with two perpendicular diameters intersecting at the center of the disk. He draws 25 more line segments through the disk, drawing each segment by selecting two points at random on the perimeter of the disk in different quadrants and connecting those two points. Find the expected number of regions into which these 27 line segments divide the disk."    },
]
]

pretrained_model_name = "togethercomputer/M1-3B"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained_model_name, torch_dtype=torch.float16).cuda().eval()

formatted_prompts = [
    tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages
]

prompts = [
    tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=200)
    for formatted_prompt in formatted_prompts
]

max_len = 8000
padded_inputs = [F.pad(t, (max_len - t.size(1), 0), value=tokenizer.pad_token_id) for t in prompts]

batch_prompts = torch.cat(padded_inputs, dim=0).cuda()

outs = model.generate(
    input_ids=batch_prompts,
    max_new_tokens=500,
    top_k=1,
    top_p=1,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    output_scores=False,
    return_dict_in_generate=False,
)

# detokenise & show
for p, o in zip(formatted_prompts, outs):
    txt = tokenizer.decode(o.tolist(), skip_special_tokens=True)
    print("═" * 80)
    print(f"PROMPT: {p}\n")
    print(txt)


outs = model.generate(
    input_ids=batch_prompts,
    max_new_tokens=500,
    top_k=1,
    top_p=1,
    temperature=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    output_scores=False,
    return_dict_in_generate=False,
)

# detokenise & show
for p, o in zip(formatted_prompts, outs):
    txt = tokenizer.decode(o.tolist(), skip_special_tokens=True)
    print("═" * 80)
    print(f"PROMPT: {p}\n")
    print(txt)