import os
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class SkillDefinition(BaseModel):
    skill_id: str
    name: str
    namespace: str
    description: str
    keywords: List[str] = Field(default_factory=list)
    file_patterns: List[str] = Field(default_factory=list)
    exemplars: List[str] = Field(default_factory=list)
    allowed_tiers: List[str] = Field(default_factory=lambda: ["free", "premium", "scholar", "enterprise"])
    summaries: Dict[str, str] = Field(default_factory=dict)  # map: doc_title -> summary (<1500 tokens)

class SkillNamespaceRegistry:
    """
    skills.ms Runtime Properties Registry:
    Loads skill definitions and metadata summaries dynamically at runtime from properties/JSON configuration.
    """
    def __init__(self, properties_file: Optional[str] = "skills_registry.json"):
        self.properties_file = properties_file
        self._skills: Dict[str, SkillDefinition] = {}
        
        if properties_file and os.path.exists(properties_file):
            self.load_from_properties(properties_file)
        else:
            self._bootstrap_default_skills()
            if properties_file:
                self.save_to_properties(properties_file)

    def _bootstrap_default_skills(self):
        default_skills = [
            SkillDefinition(
                skill_id="culinary_recipes",
                name="Culinary & Recipes",
                namespace="recipes_culinary",
                description="Cooking recipes, culinary techniques, ingredients, dietary guides, meal preparation.",
                keywords=["recipe", "cook", "ingredients", "bake", "fry", "boil", "seasoning", "dish", "cuisine", "taste", "calories", "protein", "chef", "sauce"],
                file_patterns=["*recipe*", "*culinary*", "*food*", "*meal*", "*cooking*"],
                exemplars=[
                    "how to cook crispy chicken breast in an air fryer",
                    "substitute for heavy cream in pasta carbonara",
                    "oven baking temperature and cook time for sourdough",
                    "authentic italian lasagna meat sauce recipe",
                    "best marinade and seasoning for grilled salmon"
                ],
                allowed_tiers=["free", "premium", "scholar", "enterprise"]
            ),
            SkillDefinition(
                skill_id="appliance_care",
                name="Appliance Troubleshooting & Specs",
                namespace="appliances_troubleshooting",
                description="Manuals, error codes, warranty, electrical specs, repair guides for kitchen appliances.",
                keywords=["manual", "error", "troubleshoot", "repair", "voltage", "watts", "microwave", "oven", "air fryer", "blender", "dishwasher", "refrigerator", "warranty", "reset", "filter"],
                file_patterns=["*manual*", "*appliance*", "*error*", "*spec*", "*troubleshoot*", "*user_guide*"],
                exemplars=[
                    "error code E3 flashing on instant pot display",
                    "microwave heating element not working turns off",
                    "dishwasher leaking water from bottom door seal",
                    "refrigerator compressor running warm wattage rating",
                    "how to reset thermal fuse on blender motor"
                ],
                allowed_tiers=["premium", "scholar", "enterprise"]
            ),
            SkillDefinition(
                skill_id="home_decor",
                name="Home Decor & Kitchen Design",
                namespace="home_decor_design",
                description="Kitchen design layouts, cabinetry, countertop aesthetics, interior lighting, storage organization.",
                keywords=["decor", "design", "layout", "cabinet", "countertop", "aesthetic", "color", "marble", "granite", "backsplash", "lighting", "island", "shelving"],
                file_patterns=["*decor*", "*design*", "*layout*", "*aesthetic*", "*interior*"],
                exemplars=[
                    "modern Scandinavian kitchen cabinet color palette",
                    "countertop overhang dimensions for kitchen island seating",
                    "under-cabinet LED lighting placement layout ideas",
                    "quartz vs granite countertop aesthetic comparison",
                    "small kitchen pantry organization shelving layout"
                ],
                allowed_tiers=["premium", "scholar", "enterprise"]
            ),
            SkillDefinition(
                skill_id="cleaning_maintenance",
                name="Cleaning & Maintenance",
                namespace="cleaning_maintenance",
                description="Sanitization techniques, stain removal, countertop care, plumbing, appliance deep cleaning.",
                keywords=["clean", "cleaning", "stain", "sanitize", "soap", "vinegar", "bleach", "scrub", "rust", "maintenance", "plumbing", "drain", "mold", "grease", "seal", "countertop", "granite", "marble", "care"],
                file_patterns=["*clean*", "*maintenance*", "*stain*", "*care*", "*sanitize*"],
                exemplars=[
                    "remove stubborn red wine stain from granite countertop",
                    "deep clean oven interior with baking soda and vinegar",
                    "unclog kitchen sink drain garbage disposal maintenance",
                    "how to seal natural stone marble countertops",
                    "descaling electric kettle and coffee maker with citric acid"
                ],
                allowed_tiers=["premium", "scholar", "enterprise"]
            ),
            SkillDefinition(
                skill_id="academia_research",
                name="Academic Research & Science",
                namespace="academia_research",
                description="Peer-reviewed research papers, methodologies, empirical evaluations, neural architectures.",
                keywords=["research", "paper", "methodology", "empirical", "evaluation", "architecture", "dataset", "citations", "latex", "theorem", "transformer"],
                file_patterns=["*paper*", "*research*", "*study*", "*thesis*"],
                exemplars=[
                    "neural network transformer architecture attention mechanism",
                    "empirical comparative analysis of retrieval augmented generation",
                    "peer-reviewed evaluation metrics for domain classification",
                    "statistical significance testing in large language model benchmarks",
                    "loss convergence mathematical formulation in deep learning"
                ],
                allowed_tiers=["scholar", "enterprise"]
            ),
        ]
        for skill in default_skills:
            self.register_skill(skill)

    def register_skill(self, skill: SkillDefinition) -> None:
        self._skills[skill.skill_id] = skill

    def update_dynamic_summary(self, namespace: str, document_title: str, summary: str, extra_keywords: List[str]) -> None:
        """
        Dynamically updates the skills.ms meta store at runtime with document summary (<1500 tokens).
        Routing keywords are kept clean and unpolluted to maintain domain routing integrity.
        """
        target_skill = next((s for s in self._skills.values() if s.namespace == namespace), None)
        
        if not target_skill:
            skill_id = f"dynamic_{namespace}"
            target_skill = SkillDefinition(
                skill_id=skill_id,
                name=namespace.replace("_", " ").title(),
                namespace=namespace,
                description=f"Dynamic domain for {namespace}",
                keywords=list(extra_keywords),
                exemplars=[document_title] if document_title else []
            )
            self._skills[skill_id] = target_skill

        # Store document summary strictly without mutating canonical routing keywords
        target_skill.summaries[document_title] = summary

        if self.properties_file:
            self.save_to_properties(self.properties_file)

    def get_skill(self, skill_id: str) -> Optional[SkillDefinition]:
        return self._skills.get(skill_id)

    def get_all_namespaces(self) -> List[str]:
        return list(set(skill.namespace for skill in self._skills.values()))

    def list_skills(self) -> List[SkillDefinition]:
        return list(self._skills.values())

    def save_to_properties(self, file_path: str) -> None:
        data = [skill.model_dump() for skill in self._skills.values()]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_properties(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._skills = {item["skill_id"]: SkillDefinition(**item) for item in data}
