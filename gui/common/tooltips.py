"""
Centralized tooltip definitions for all GUIs.

Tooltip text for parameters, checkboxes, and UI elements used across
Individual Simulation, Gradient Descent Optimization, and Batch Runner.
"""

# Main simulation parameter tooltips
# Keys match the parameter names in the GUI
PARAMETER_TOOLTIPS = {
    "Random Seed (optional)": (
        "Random Seed (Integer)\n\n"
        "Sets the random number generator seed for reproducibility.\n"
        "• Same seed → identical simulation results\n"
        "• Different seeds → different evolutionary trajectories\n"
        "• Leave blank for random seed\n\n"
        "Used for: initial population sampling, mutations, death/duplication events"
    ),
    
    "Cost of Life": (
        "Cost of Life (Energy/generation)\n\n"
        "Fixed energy cost deducted each generation for basic maintenance.\n\n"
        "Equation:\n"
        "  Energy(t+1) = Energy(t) - cost_of_life\n\n"
        "• Higher values → organisms need more energy production to survive\n"
        "• 0 → no maintenance cost (only enzyme production costs)\n"
        "• Encourages efficiency and resource acquisition"
    ),
    
    "Mutation Rate": (
        "Mutation Rate (Probability)\n\n"
        "Probability that each enzyme in an offspring mutates.\n\n"
        "Applied independently to each of the 3 enzymes (A, B, T).\n\n"
        "Process:\n"
        "  1. For each enzyme: if random() < mutation_rate:\n"
        "  2.   enzyme += random_normal(0, mutation_scale)\n"
        "  3.   enzyme = clip(enzyme, 0, ∞)\n\n"
        "• Typical values: 0.001 - 0.1\n"
        "• Higher values → more genetic variation but less stable adaptation\n"
        "• 0 → no evolution (clonal reproduction only)"
    ),
    
    "Mutation Scale": (
        "Mutation Scale (Standard Deviation)\n\n"
        "Standard deviation of the normal distribution used for mutations.\n"
        "Determines the typical size of mutational changes.\n\n"
        "Equation:\n"
        "  Δenzyme ~ Normal(mean=0, std=mutation_scale)\n\n"
        "• Small values (0.01-0.05) → fine-tuning, slow evolution\n"
        "• Large values (0.2-0.5) → large jumps, fast exploration\n"
        "• Combined with mutation_rate to control evolutionary dynamics\n\n"
        "Example: scale=0.1 means ~68% of mutations change enzyme by ±0.1"
    ),
    
    "Number of Generations": (
        "Number of Generations (Integer)\n\n"
        "Total number of simulation timesteps to run.\n\n"
        "Each generation:\n"
        "  1. Environmental inflow: M1 and M2 added to environment\n"
        "  2. M1 import: Organisms import M1 from environment via Enzyme T\n"
        "     (facilitated diffusion with saturation kinetics)\n"
        "  3. M1 → M2 conversion: Stored M1 converted to M2 using Enzyme A (Task A)\n"
        "  4. M2 diffusion & consumption: M2 passively diffuses out (simple diffusion)\n"
        "     AND internal M2 consumed → generates energy via Enzyme B (Task B)\n"
        "     (these processes compete for the same stored M2 pool)\n"
        "  5. Energy costs deducted (enzyme production/maintenance costs)\n"
        "  6. Death check (low energy → higher death probability)\n"
        "  7. Duplication check (high energy → higher duplication probability)\n"
        "  8. Offspring mutate\n\n"
        "• More generations → longer evolutionary time\n"
        "• Typical: 500-2000 for observable evolution"
    ),
    
    "Initial Organism Count": (
        "Initial Organism Count (Integer)\n\n"
        "Number of organisms at the start of the simulation.\n\n"
        "• Population size fluctuates based on death and duplication\n"
        "• If homogeneous mode: all start with same genotype\n"
        "• Otherwise: sampled uniformly from [0, 1]\n\n"
        "Notes:\n"
        "• Larger populations → more genetic diversity, slower computation\n"
        "• Small populations → genetic drift dominates, faster extinction\n"
        "• Typical: 50-200 organisms"
    ),
    
    "Average In-Flow": (
        "Average In-Flow (M1 per generation)\n\n"
        "Amount of metabolite M1 added to environment each generation.\n"
        "This is the system's energy input.\n\n"
        "Equation:\n"
        "  M1_env(t+1) = M1_env(t) + average_inflow - Σ(consumed_M1)\n\n"
        "M1 is continuously supplied and consumed by organisms (Task A).\n\n"
        "• Higher inflow → more resources → larger populations\n"
        "• Lower inflow → resource limitation → competition\n"
        "• Must balance with population size and costs"
    ),

    "Average In_Flow (Acetate)": (
        "Average In-Flow (Acetate) (M2 per generation)\n\n"
        "Amount of acetate (M2) added directly to the environment each generation\n"
        "when 'Enable Acetate Addition' is turned on.\n\n"
        "Equation:\n"
        "  M2_env(t+1) = M2_env(t) + acetate_inflow - uptake/export effects\n\n"
        "• Higher values increase external acetate availability\n"
        "• 0 means no additional acetate input\n"
        "• Only used when acetate-addition toggle is enabled"
    ),
    
    "Initial Energy": (
        "Initial Energy (Per Organism)\n\n"
        "Starting internal energy assigned to each organism when the\n"
        "'Enable Initial Energy' setting is active.\n\n"
        "• Higher values can delay early deaths and boost initial growth\n"
        "• 0 starts organisms without initial stored energy\n"
        "• Ignored when 'Enable Initial Energy' is disabled"
    ),
    
    "Duplication Sigmoid Midpoint": (
        "Duplication Sigmoid Midpoint (Energy)\n\n"
        "Energy level at which duplication probability = 50%.\n"
        "This is an ABSOLUTE energy value, not multiplied by cost_of_life.\n\n"
        "Equation:\n"
        "  P(duplicate) = 1 / (1 + exp(-intensity × (E - midpoint)))\n\n"
        "At E = midpoint: P = 0.5\n"
        "At E = midpoint + δ: P increases (steepness controlled by intensity)\n\n"
        "• Low values (0.5-2) → easy reproduction, rapid growth\n"
        "• High values (5-10) → strict energy requirement, slow growth\n"
        "• Should consider enzyme costs and inflow when setting"
    ),
    
    "Duplication Sigmoid Intensity": (
        "Duplication Sigmoid Intensity (1/Energy)\n\n"
        "Controls the steepness of the duplication probability curve.\n\n"
        "Equation:\n"
        "  P(duplicate) = 1 / (1 + exp(-k × (E - midpoint)))\n"
        "where k = intensity\n\n"
        "• Low values (0-2) → gradual transition, wide range of duplication energies\n"
        "• Moderate values (3-7) → steeper transition around the midpoint\n"
        "• High values (8-10) → near-threshold behavior around the midpoint\n\n"
        "Default range: 0 to 10 (default 5)\n"
        "Derivative at midpoint = intensity/4, so intensity=10 gives slope=2.5"
    ),
    
    "M1 Facilitated Diffusion Constant": (
        "M1 Facilitated Diffusion Constant (1/time)\n\n"
        "Diffusion coefficient for M1 (glucose) facilitated diffusion.\n"
        "Used in the Michaelis-Menten saturation kinetics for M1 import via Enzyme T.\n\n"
        "Equation:\n"
        "  Import Flux = enzyme_T × M1_diffusion_constant × cell_volume × (gradient / (K_m + gradient))\n"
        "  where gradient = [M1]_external - [M1]_internal\n\n"
        "• Higher values → faster M1 import (for same gradient and enzyme_T)\n"
        "• Lower values → slower M1 import\n"
        "• Controls the maximum transport rate (V_max) when saturation occurs\n\n"
        "Key differences from M2 simple diffusion:\n"
        "  - M1 uses facilitated diffusion with saturation (Michaelis-Menten)\n"
        "  - M1 requires Enzyme T to function\n"
        "  - M1 flux saturates at high gradients\n"
        "  - This constant is independent of M2 simple diffusion constant\n\n"
        "Typical: 0.1-10"
    ),
    
    "M2 Simple Diffusion Constant": (
        "M2 Simple Diffusion Constant (1/time)\n\n"
        "Diffusion coefficient for M2 (acetate/glycerol) simple diffusion.\n"
        "Used in Fick's Law for M2 passive diffusion (bidirectional).\n\n"
        "Equation:\n"
        "  Diffusion Flux = ([M2]_internal - [M2]_external) × M2_diffusion_constant × cell_volume\n"
        "  Positive flux (internal > external) = export (M2 diffuses out)\n"
        "  Negative flux (external > internal) = import (M2 diffuses in)\n\n"
        "• Higher values → faster M2 diffusion (for same concentration gradient)\n"
        "• Lower values → slower M2 diffusion\n"
        "• Linear relationship with concentration gradient (no saturation)\n"
        "• Bidirectional: M2 can diffuse both in and out depending on gradient\n\n"
        "This constant is independent of M1 facilitated diffusion constant.\n\n"
        "Typical: 0.1-10"
    ),
    
    "M1 Saturation Constant": (
        "M1 Saturation Constant (K_m, Concentration Units)\n\n"
        "Michaelis-Menten saturation constant for M1 facilitated diffusion.\n"
        "Controls when transport rate saturates at high concentration gradients.\n\n"
        "Equation:\n"
        "  Saturation term = gradient / (K_m + gradient)\n"
        "  where gradient = [M1]_external - [M1]_internal\n"
        "  At low gradients (gradient << K_m): flux ≈ linear with gradient\n"
        "  At high gradients (gradient >> K_m): flux saturates at V_max\n"
        "  V_max = enzyme_T × M1_facilitated_diffusion_constant × cell_volume\n\n"
        "• Low values (0.01-0.1) → early saturation, transport saturates quickly\n"
        "• High values (1-10) → late saturation, transport remains linear longer\n"
        "• Very high values → no saturation (approaches linear)\n\n"
        "Biological interpretation:\n"
        "  Transporter proteins have limited capacity.\n"
        "  At high substrate concentrations, all transporters are occupied.\n"
        "  K_m is the gradient at which transport is at half-maximum rate.\n\n"
        "Typical: 0.1-1.0"
    ),

    "Cost of Transport": (
        "Cost of Transport (Energy/generation)\n\n"
        "Per-generation energy cost for maintaining M1 facilitation.\n"
        "Applied per organism as:\n"
        "  cost = cost_of_transport × facilitation_trait\n\n"
        "• Higher values penalize strong facilitation\n"
        "• 0 disables the cost\n"
        "• Only used when M1 facilitated diffusion is enabled"
    ),

    "Initial Facilitation": (
        "Initial Facilitation (Trait value)\n\n"
        "Initial value of the M1 facilitation trait when homogeneous mode is enabled.\n"
        "Trait ranges from 0 to 1.\n\n"
        "• 0 → no facilitation\n"
        "• 1 → maximal facilitation\n"
        "• Only used when M1 facilitated diffusion is enabled"
    ),
    
    "Chemostat Volume": (
        "Chemostat Volume (Volume units, e.g., mL)\n\n"
        "Shown only when M1 and/or M2 diffusion is enabled.\n"
        "Converts environmental pool amounts to concentrations for diffusion fluxes:\n"
        "  [M1]_env = M1_amount / V_eff\n"
        "  [M2]_env = M2_amount / V_eff\n"
        "(V_eff accounts for biomass displacement.)\n\n"
        "Also caps duplication by spare volume when diffusion limits biomass.\n"
        "Unused in the pooled (no-diffusion) regime.\n\n"
        "Typical: 10000–20000 in primary batches"
    ),

    "Flow Percentage": (
        "Flow Percentage (% per generation)\n\n"
        "When Chemostat Flow is enabled, each organism is removed independently with\n"
        "this probability, and environmental M1/M2 pools are multiplied by (1 - phi).\n\n"
        "Example: 40% flow removes ~40% of organisms (Bernoulli) and 40% of env metabolites.\n\n"
        "Typical: 0-20"
    ),

    "Intermediate Costs": (
        "Intermediate Costs (Energy per internal M2 per generation)\n\n"
        "When Intermediate Costs are enabled, organisms pay an energetic penalty\n"
        "proportional to their stored internal M2 each generation.\n\n"
        "Equation:\n"
        "  Energy(t+1) = Energy(t) - intermediate_costs × stored_M2(t)\n\n"
        "• Higher values penalize hoarding of internal storage\n"
        "• 0 disables the penalty\n"
        "• Typical: 0-1"
    ),
    
    "Cell Volume": (
        "Cell Volume (Volume units, e.g., mL)\n\n"
        "Volume of a single organism cell.\n"
        "Used to calculate internal metabolite concentrations.\n\n"
        "Internal concentrations:\n"
        "  [M1]_internal = stored_M1 / cell_volume\n"
        "  [M2]_internal = stored_M2 / cell_volume\n\n"
        "Effects:\n"
        "• Larger cells → lower internal concentrations (for same amount)\n"
        "  → smaller M2 diffusion gradient\n"
        "  → less M2 export\n"
        "• Smaller cells → higher internal concentrations\n"
        "  → larger diffusion gradients\n"
        "  → more M2 export\n\n"
        "Typical: 1e-6 to 1e-9 (micro to nanoliters)\n"
        "Default: 1e-6 (1 microliter)"
    ),
    
    "Intermediate Storage Cost": (
        "Intermediate Storage Cost (Energy per M2 per generation)\n\n"
        "Cost per unit of stored internal M2 each generation.\n"
        "Penalizes hoarding of metabolites.\n\n"
        "Equation:\n"
        "  Energy(t+1) = Energy(t) - cost × stored_M2(t)\n\n"
        "Applied BEFORE duplication/death checks.\n\n"
        "• Higher values → organisms penalized for storing M2\n"
        "  → encourages rapid M2 consumption (Task B)\n"
        "  → or export M2 (diffusion) to avoid cost\n"
        "• 0 → no storage penalty\n"
        "• Typical: 0.01-0.2"
    ),
    
    "Degradation Rate": (
        "Degradation Rate (Fraction per generation)\n\n"
        "Fraction of ALL metabolites that degrade (disappear) each generation.\n"
        "Applies uniformly to:\n"
        "  • Environmental M1 (metab1_env)\n"
        "  • Environmental M2 (metab2_env)\n"
        "  • Stored M1 inside organisms (stored_M1)\n"
        "  • Stored M2 inside organisms (stored_M2)\n\n"
        "Applied after all metabolic processes and transport.\n\n"
        "Equation (for each metabolite):\n"
        "  metabolite(t+1) = metabolite(t) × (1 - degradation_rate)\n\n"
        "• degradation_rate = 0.1 → 10% of all metabolites degrade each generation\n"
        "• degradation_rate = 0 → no degradation (default)\n"
        "• degradation_rate = 1 → all metabolites degrade completely\n\n"
        "Effects:\n"
        "  • Higher values → metabolites don't accumulate\n"
        "    → organisms must use them quickly\n"
        "    → reduces cross-feeding opportunity\n"
        "  • Lower values → metabolites persist longer\n"
        "    → allows storage and accumulation\n\n"
        "Similar to storage cost but removes metabolite rather than costing energy.\n\n"
        "Typical: 0-0.2"
    ),
    
    "Death Decay Rate": (
        "Death Decay Rate (1/Energy)\n\n"
        "Controls how death probability decreases with increasing energy.\n"
        "Uses an exponential decay model (ignored when Binary Death at Zero Energy or Constant Death Probability is on).\n\n"
        "Equation:\n"
        "  P(death) = exp(-rate × energy)\n\n"
        "At energy = 0: P(death) = 100%\n"
        "At energy = 1/rate: P(death) ≈ 36.8%\n"
        "At energy = 3/rate: P(death) ≈ 5%\n\n"
        "• Higher rate → death probability drops faster with energy\n"
        "  → forgiving to organisms with modest energy\n"
        "• Lower rate → death remains likely even at moderate energy\n"
        "  → harsh selection for high energy\n\n"
        "Typical: 1-20\n"
        "Example: rate=10 means P(death)<5% when energy>0.3"
    ),

    "Constant Probability": (
        "Constant Probability\n\n"
        "Flat per-generation rate when Constant Death Probability and/or\n"
        "Constant Duplication Probability is enabled.\n\n"
        "Equation:\n"
        "  P(death) = this value   when constant death is on\n"
        "  P(dup) = this value     when constant duplication is on\n\n"
        "When both constant toggles are on, the same value applies to both\n"
        "(avoids exponential population drift from mismatched rates).\n\n"
        "Typical neutral reference: 0.5"
    ),
    
    "Enzyme A Cost": (
        "Enzyme A Cost (Energy per unit per generation)\n\n"
        "Energy cost for producing and maintaining enzyme A.\n"
        "See Enzyme Costs tooltip for full details."
    ),
    
    "Enzyme B Cost": (
        "Enzyme B Cost (Energy per unit per generation)\n\n"
        "Energy cost for producing and maintaining enzyme B.\n"
        "See Enzyme Costs tooltip for full details."
    ),
    
    "Enzyme T Cost": (
        "Enzyme T Cost (Energy per unit per generation)\n\n"
        "Energy cost for producing and maintaining enzyme T (M1 transporter).\n\n"
        "Enzyme T function:\n"
        "  - Facilitates M1 (glucose) import via facilitated diffusion\n"
        "  - Import rate = enzyme_T × M1_facilitated_diffusion_constant × cell_volume × (gradient / (K_m + gradient))\n"
        "  - Higher enzyme_T → faster M1 import (up to saturation)\n"
        "  - Required for M1 import (no import if enzyme_T = 0)\n\n"
        "See Enzyme Costs tooltip for full details on cost mechanics."
    ),

    "Acetate Ratio": (
        "Acetate Ratio (dimensionless)\n\n"
        "Multiplier for the ATP/energy yield from using acetate (M2) relative to glucose (M1).\n\n"
        "Energy gain equation:\n"
        "  energy_gain = A_share + acetate_ratio × B_share\n\n"
        "Interpretation:\n"
        "• acetate_ratio = 1.0 → acetate yields the same energy as glucose\n"
        "• acetate_ratio > 1.0 → acetate yields MORE energy (extra ATP)\n"
        "• acetate_ratio < 1.0 → acetate yields LESS energy\n\n"
        "Typical: 0.5–5 (choose based on your biological assumption)"
    ),

    "Investment Modifier": (
        "Investment Modifier (exponent)\n\n"
        "Exponent applied to enzyme traits when computing task investment:\n"
        "  Afunc = A^inv_mod,  Bfunc = B^inv_mod\n\n"
        "• inv_mod = 1 → linear investment in enzyme level\n"
        "• inv_mod > 1 → diminishing returns (high traits cost more to use)\n"
        "• inv_mod < 1 → accelerating returns\n\n"
        "Typical: 0.5–2.0"
    ),

    "Diffusion Constant": (
        "Diffusion Constant (1/time scale)\n\n"
        "Base diffusion rate for M1/M2 transport when diffusion toggles are on.\n"
        "Per-organism diffusion traits scale this constant (unless homogeneous\n"
        "initial diffusion is enabled).\n\n"
        "Shown in the GUI only when M1 and/or M2 diffusion is enabled."
    ),

    "Initial A": (
        "Initial A (trait, 0–1)\n\n"
        "Starting glycolysis-style enzyme trait (Task A) for each organism.\n"
        "• Homogeneous population: all organisms share this value\n"
        "• Otherwise: sampled uniformly in [0, 1] unless fixed here"
    ),

    "Initial B": (
        "Initial B (trait, 0–1)\n\n"
        "Starting TCA-style enzyme trait (Task B) for each organism.\n"
        "• With coupled traits: B is often 1 − A unless Independent A/B Traits is on\n"
        "• Homogeneous population: all organisms share this value"
    ),
}


# Specialized tooltips for UI elements

ENZYME_COST_TOOLTIP = (
    "Enzyme Costs (Energy per unit per generation)\n\n"
    "Energy cost for producing and maintaining each unit of enzyme.\n\n"
    "Equation:\n"
    "  Total_Cost = cost_A × enzyme_A + cost_B × enzyme_B + cost_T × enzyme_T\n"
    "  Energy(t+1) = Energy(t) - Total_Cost\n\n"
    "Applied each generation before death/duplication checks.\n\n"
    "Evolutionary pressure:\n"
    "• Higher cost → selection for lower enzyme values\n"
    "• Lower cost → enzymes can evolve to higher values\n"
    "• Different costs → trade-offs between enzymes\n\n"
    "Example scenarios:\n"
    "• High cost_A (0.5): Production is expensive → favor importers\n"
    "• High cost_B (0.5): Consumption is expensive → favor exporters\n"
    "• High cost_T (0.5): Transport is expensive → favor self-sufficiency\n"
    "• Low cost_B (0.05): Cheap consumption → favor rapid M2 processing\n\n"
    "Typical: 0.05-0.3 per enzyme"
)

HOMOGENEOUS_TOOLTIP = (
    "Homogeneous Population Mode\n\n"
    "Start all organisms with identical genotype instead of sampling from ranges.\n\n"
    "When ENABLED:\n"
    "• All organisms start with the specified Initial A, B, T values\n"
    "• Diversity arises purely from mutations during simulation\n"
    "• Useful for studying evolution from a specific starting point\n\n"
    "When DISABLED:\n"
    "• Organisms start with diverse genotypes\n"
    "• Initial enzymes sampled uniformly from [0, 1]\n"
    "• Allows exploring evolution from different initial conditions\n\n"
    "Use cases for homogeneous mode:\n"
    "• Testing if a specific genotype is evolutionarily stable\n"
    "• Studying adaptation from a known starting configuration\n"
    "• Comparing outcomes of identical starting populations\n"
    "• Genotype sweeps (systematically testing different starting genotypes)"
)

SIMULATION_SETTINGS_TOOLTIPS = {
    "Silent Mode": (
        "Silent Mode\n\n"
        "Suppresses progress printing and non-critical simulation warnings.\n"
        "Useful for faster, cleaner batch runs and optimization."
    ),
    "Enable Initial Energy": (
        "Enable Initial Energy\n\n"
        "If enabled, each organism starts with the specified Initial Energy value.\n"
        "If disabled, initial energy is forced to 0.0."
    ),
    "Enable Chemostat Flow": (
        "Enable Chemostat Flow\n\n"
        "When enabled, a percentage of chemostat volume is removed each generation.\n"
        "This removes organisms and environmental metabolites proportionally."
    ),
    "Enable Intermediate Costs": (
        "Enable Intermediate Costs\n\n"
        "Applies an energy penalty proportional to stored internal M2 each generation.\n"
        "Uses the 'Intermediate Costs' parameter value."
    ),
    "Enable Acetate Addition": (
        "Enable Acetate Addition\n\n"
        "Adds acetate (M2) to the environment each generation.\n"
        "Uses the 'Average In_Flow (Acetate)' parameter value."
    ),
    "Binary Death at Zero Energy": (
        "Binary Death at Zero Energy\n\n"
        "Neutral-style death rule: P(death)=1 if energy ≤ 0, else P(death)=0.\n"
        "Replaces the exponential death curve (Death Decay Rate is ignored).\n\n"
        "With Constant Death Probability also on: still forces P(death)=1 at energy ≤ 0;\n"
        "above zero uses Constant Probability instead of 0."
    ),
    "No Death": (
        "No Death\n\n"
        "Forces death probability to 0 at all energies.\n"
        "Death Decay Rate and Binary Death behavior are ignored while enabled.\n\n"
        "Special case with Constant Duplication Probability:\n"
        "Constant Probability is ignored and duplication is derived from\n"
        "chemostat flow using p_dup = phi/(1-phi), where phi=Flow Percentage/100,\n"
        "clipped to [0,1] for simulation stability."
    ),
    "Constant Death Probability": (
        "Constant Death Probability\n\n"
        "Flat death rate P(death) = Constant Probability at all energies.\n"
        "Ignores Death Decay Rate.\n\n"
        "Combined with Binary Death at Zero Energy: P(death)=1 at energy ≤ 0,\n"
        "Constant Probability above zero."
    ),
    "Constant Duplication Probability": (
        "Constant Duplication Probability\n\n"
        "Flat duplication rate P(dup) = Constant Probability at all energies.\n"
        "Ignores sigmoid midpoint/intensity.\n\n"
        "With No Death: requires Enable Chemostat Flow; duplication is then\n"
        "derived from flow (phi/(1-phi)) and Constant Probability is ignored.\n"
        "Without flow, this toggle cannot stay on while No Death is enabled."
    ),
}

SKIP_ANIMATION_TOOLTIP = (
    "Skip Animation (Recommended)\n\n"
    "When enabled, jumps directly to the final generation instead of animating.\n\n"
    "Benefits:\n"
    "• Much faster (10-100x speedup for long simulations)\n"
    "• Lower memory usage\n"
    "• Avoids GUI lag during animation\n\n"
    "Trade-off:\n"
    "• Cannot observe evolutionary dynamics in real-time\n"
    "• All plots shown at final generation only\n\n"
    "Recommendation: Keep enabled unless you specifically want to watch evolution unfold."
)

# Gradient Descent Optimization tooltips
GRADIENT_DESCENT_TOOLTIPS = {
    "Learning Rate": (
        "Learning Rate (Step Size)\n\n"
        "Controls how large steps the optimizer takes in parameter space.\n\n"
        "Equation:\n"
        "  new_param = old_param + learning_rate × gradient\n\n"
        "• Higher values (0.1-1.0) → larger steps, faster convergence, but may overshoot\n"
        "• Lower values (0.001-0.01) → smaller steps, more stable, but slower convergence\n"
        "• Too high → may diverge or oscillate around optimum\n"
        "• Too low → very slow convergence\n\n"
        "Typical: 0.01-0.1\n"
        "Default: 0.01"
    ),
    
    "Max Iterations": (
        "Max Iterations\n\n"
        "Maximum number of gradient descent steps to perform.\n\n"
        "Each iteration:\n"
        "  1. Evaluate metric at current parameters\n"
        "  2. Compute gradient (numerical differentiation)\n"
        "  3. Update parameters: param += learning_rate × gradient\n"
        "  4. Check convergence\n\n"
        "• More iterations → better chance of finding optimum, but slower\n"
        "• Fewer iterations → faster, but may stop before convergence\n"
        "• Optimization stops early if convergence threshold is met\n\n"
        "Typical: 10-50\n"
        "Default: 20"
    ),
    
    "Convergence Threshold": (
        "Convergence Threshold\n\n"
        "Stops optimization when parameter changes become very small.\n\n"
        "Convergence check:\n"
        "  If |param_change| < threshold for all parameters:\n"
        "    Stop optimization (converged)\n\n"
        "• Smaller values (0.0001) → stricter convergence, more precise results\n"
        "• Larger values (0.01) → looser convergence, faster stopping\n"
        "• Too small → may never converge due to numerical noise\n"
        "• Too large → may stop before reaching optimum\n\n"
        "Typical: 0.001-0.01\n"
        "Default: 0.001"
    ),
    
    "Gradient Step Size": (
        "Gradient Step Size (Numerical Differentiation)\n\n"
        "Step size used to compute gradients numerically.\n\n"
        "Gradient computation:\n"
        "  gradient ≈ (metric(param + step) - metric(param)) / step\n\n"
        "• Smaller values (0.001-0.01) → more accurate gradients, but sensitive to noise\n"
        "• Larger values (0.1) → less accurate, but more robust to noise\n"
        "• Should be small relative to parameter values\n"
        "• Too small → numerical errors dominate\n"
        "• Too large → poor gradient approximation\n\n"
        "Typical: 0.01-0.1\n"
        "Default: 0.01"
    ),
    
    "Number of Random Starts": (
        "Number of Random Starts\n\n"
        "Number of different random starting points in parameter space.\n\n"
        "Process:\n"
        "  1. Generate N random parameter sets within bounds\n"
        "  2. Run full gradient descent from each starting point\n"
        "  3. Track best result across all runs\n\n"
        "Benefits:\n"
        "• Explores different regions of parameter space\n"
        "• Reduces chance of getting stuck in local optima\n"
        "• Accounts for stochasticity in simulations\n\n"
        "• More starts → better exploration, but slower\n"
        "• Fewer starts → faster, but may miss global optimum\n\n"
        "Typical: 4-20\n"
        "Default: 4"
    ),
    
    "Gradient Descents per Start": (
        "Gradient Descents per Start\n\n"
        "Number of independent gradient descents to run from each random starting point.\n\n"
        "Total runs = Number of Random Starts × Gradient Descents per Start\n\n"
        "Why multiple descents per start?\n"
        "• Simulations are stochastic (different seeds give different results)\n"
        "• Running multiple descents with different seeds accounts for this variability\n"
        "• Each descent uses a different random seed offset\n\n"
        "Example:\n"
        "  • 4 random starts × 10 descents per start = 40 total gradient descents\n"
        "  • Each descent explores the same starting point but with different simulation randomness\n\n"
        "Typical: 1-10\n"
        "Default: 1"
    ),
    
    "Metric to Optimize": (
        "Metric to Optimize\n\n"
        "The simulation metric to maximize or minimize.\n\n"
        "Available metrics:\n"
        "• Final Population Size, task-share summaries, trait entropy, neutral percentiles, etc.\n"
        "• See the metric dropdown for the current list (SIMULATION_METRIC_NAMES).\n\n"
        "The optimizer will adjust parameters to find the best value of this metric."
    ),
    
    "Optimization Goal": (
        "Optimization Goal\n\n"
        "Whether to maximize or minimize the selected metric.\n\n"
        "• Maximize: Find parameters that give highest metric value\n"
        "• Minimize: Find parameters that give lowest metric value\n\n"
        "Example:\n"
        "  • Maximize Task2 Share Weighted Prob. Mean → bias toward Task-2 share\n"
        "  • Minimize Trait Std Dev (Coupled) → reduce trait diversity"
    ),
    
    "Number of Replicates": (
        "Number of Replicates\n\n"
        "Number of independent simulations to run at each parameter set.\n\n"
        "Process:\n"
        "  1. For each gradient descent step:\n"
        "  2.   Run N independent simulations (different random seeds)\n"
        "  3.   Compute metric for each simulation\n"
        "  4.   Average the metrics to get final metric value\n\n"
        "Benefits:\n"
        "• Reduces noise from stochastic simulations\n"
        "• More reliable gradient estimates\n"
        "• Smoother optimization trajectory\n\n"
        "Trade-off:\n"
        "• More replicates → more accurate but slower\n"
        "• Fewer replicates → faster but noisier gradients\n\n"
        "Typical: 4-10\n"
        "Default: 1 (see Replicates field in the GUI)"
    ),
    "Fix Parameter": (
        "Fix this parameter during optimization.\n\n"
        "Fixed parameters do not receive gradient updates and are held constant."
    ),
    "Unfix Parameter": (
        "Unfix this parameter so optimization can update it.\n\n"
        "Unfixed parameters move back into the optimizable parameter set."
    ),
    "Show NaN Runs": (
        "Show NaN runs in parameter-space visualization.\n\n"
        "When enabled, failed runs are displayed; otherwise only valid runs are shown."
    ),
    "Live Plot Updates": (
        "Update plots live while optimization is running.\n\n"
        "Turning this off can improve performance for very large runs."
    ),
    "Heatmap Log Scale Axes": (
        "Use log10 scaling for heatmap axes in parameter heatmaps."
    ),
    "Heatmap Show Mean Metric": (
        "Show mean metric value per heatmap bin instead of run counts."
    ),
    "Enable M1 Diffusion (simple)": (
        "Enable simple M1 diffusion between environment and cells.\n\n"
        "Uses the global Diffusion Constant and concentration gradient.\n"
        "When enabled, organisms can import/export M1 passively (no facilitation trait required)."
    ),
    "Enable M2 Diffusion": (
        "Enable bidirectional M2 diffusion between cells and environment.\n\n"
        "When disabled, produced M2 is exported and Task B draws only from environmental M2."
    ),
    "Enable Diffusion Mutation": (
        "Enable mutation of per-organism diffusion trait D.\n\n"
        "When enabled, each offspring may mutate its D value during duplication."
    ),
    "Homogeneous Initial Diffusion Const.": (
        "Initialize per-organism diffusion trait D homogeneously from Diffusion Constant.\n\n"
        "When off, initial D values are sampled randomly.\n"
        "Only relevant when diffusion mutation is enabled."
    ),
    "Enable M1 Facilitated Diffusion": (
        "Enable facilitated M1 diffusion using trait T and transport cost.\n\n"
        "This mode is import-only (no M1 export) and is mutually exclusive with simple/porin M1 diffusion."
    ),
    "Enable M1 Porin Diffusion": (
        "Enable porin-like simple M1 diffusion (import-only).\n\n"
        "This mode uses simple diffusion for M1 import but blocks M1 export.\n"
        "Mutually exclusive with standard simple M1 diffusion and facilitated M1 diffusion."
    ),
    "Independent A/B Traits": (
        "Allow A and B traits to evolve independently.\n\n"
        "When off, traits are coupled with B = 1 - A."
    ),
}

# Batch Runner tooltips
BATCH_RUNNER_TOOLTIPS = {
    "Runs per batch": (
        "Runs per batch (N)\n\n"
        "Number of independent simulations in each batch.\n"
        "Unfixed parameters are drawn uniformly from their min/max bounds; "
        "fixed parameters and toggles stay the same for every run in the batch.\n\n"
        "Default: 1000"
    ),
    "Primary batches": (
        "Number of batches\n\n"
        "How many independent batches to run in one campaign.\n"
        "Each batch uses its own random seed and adds one hit count to the "
        "Results chart.\n\n"
        "Default: 100"
    ),
    "Random seed": (
        "Random seed (optional)\n\n"
        "Base seed for reproducibility. Leave empty to pick a seed automatically "
        "when the campaign starts."
    ),
    "Configuration name": (
        "Configuration name (optional)\n\n"
        "Label for this batch setup in CSV outputs (configuration and plot_label columns). "
        "Leave empty to infer a short name from simulation toggles (e.g. Neutral, justDup)."
    ),
    "Metric filter": (
        "Metric filter (A–D)\n\n"
        "Metric to test in this slot. Choose (none) to skip a slot.\n"
        "Active filters are combined: a run counts as a hit only when every "
        "active metric passes its operator and threshold."
    ),
    "Metric operator": (
        "Comparison operator for this metric filter.\n\n"
        "Hit when the metric value is greater than, greater than or equal to, "
        "less than, or less than or equal to the threshold."
    ),
    "Metric threshold": (
        "Threshold for this metric filter.\n\n"
        "Numeric cutoff used with the operator above."
    ),
    "Save folder": (
        "Choose Save Folder\n\n"
        "Folder where this campaign will write simulation records, a summary "
        "of the run, hit-count tables, and charts.\n\n"
        "Required before Run Batch."
    ),
    "Run batch": (
        "Run Batch\n\n"
        "Start the batch campaign. Batches run one after another.\n"
        "Use Pause / Resume, or type pause / resume in the terminal, to stop "
        "between batches."
    ),
    "Pause": "Pause after the current batch finishes.",
    "Resume": "Continue a paused campaign.",
    "Save JSON Settings": (
        "Save JSON Settings\n\n"
        "Save the current setup (batch sizes, metrics, seed, parameter bounds, "
        "and simulation toggles) to a JSON file you can reload later with "
        "Load JSON Settings."
    ),
    "Load JSON Settings": (
        "Load JSON Settings\n\n"
        "Restore batch sizes, metrics, seed, parameter bounds, and simulation "
        "toggles from a settings JSON file.\n\n"
        "Does not load finished campaign results — use Load Campaign Summary for that."
    ),
    "Load saved": (
        "Load Campaign Summary\n\n"
        "Open a finished campaign summary to view hit counts and charts, and "
        "to restore settings when they are stored in the summary.\n\n"
        "For settings-only JSON files, use Load JSON Settings instead."
    ),
}

BATCH_RERUNNER_TOOLTIPS = {
    "Session folder": (
        "Session folder\n\n"
        "Folder from a completed Batch Runner campaign — the same folder "
        "you chose when you ran the campaign."
    ),
    "Choose session folder": (
        "Choose Session Folder\n\n"
        "Pick the folder that contains a finished Batch Runner campaign."
    ),
    "Screen mode": (
        "What to re-run\n\n"
        "Hits: parameter sets that passed all metric filters in the original campaign.\n"
        "Non-hits: sets that did not pass.\n"
        "Both: run hits first, then non-hits."
    ),
    "N seeds": (
        "Seeds per point\n\n"
        "How many fresh random seeds to use for each unique parameter set (default 20).\n"
        "The summary reports what fraction of those re-runs pass the filters again."
    ),
    "Limit non-hits": (
        "Don't run all non-hits\n\n"
        "When enabled, only a sample of unique non-hits is re-screened.\n"
        "Turn off to re-run every unique non-hit (can be very large)."
    ),
    "Max non-hits": (
        "Max non-hit points\n\n"
        "How many unique non-hits to sample when the limit toggle is on "
        "(default 500)."
    ),
    "Workers": (
        "Workers\n\n"
        "Number of parallel processes used for re-simulation (default: CPU count)."
    ),
    "Dedupe": (
        "Dedupe identical parameter vectors\n\n"
        "When enabled, parameter sets that differ only in bookkeeping are "
        "counted once."
    ),
    "Quiet terminal": (
        "Quiet terminal output\n\n"
        "When enabled (default), suppresses per-point worker logs and other "
        "re-screen chatter in the terminal. Progress still appears in the GUI "
        "status line and progress bar."
    ),
    "Run": (
        "Run Re-Screen\n\n"
        "Re-run the selected parameter sets with new random seeds.\n\n"
        "Saves a summary table and JSON inside the campaign folder, showing "
        "how often each set passes the filters again."
    ),
}

