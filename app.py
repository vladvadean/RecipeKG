from transformers import T5Tokenizer, MT5ForConditionalGeneration
import torch
import torch.nn as nn
import networkx as nx
import re
import pandas as pd
import json
from torch_geometric.utils import from_networkx
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn.functional import cosine_similarity
import numpy as np
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from munkres import Munkres

# === Patterns ===
step_pattern = r"temp:\s*[^,\n]+,\s*time:\s*[^,\n]+,\s*cooking_function:\s*[^,\n]+(?:\s*OR\s*temp:\s*[^,\n]+,\s*time:\s*[^,\n]+,\s*cooking_function:\s*[^,\n]+)*"
temp_pattern = r"temp:\s*([^,\n]+)"
time_pattern = r"time:\s*([^,\n]+)"
mode_pattern = r"cooking_function:\s*(.*?)(?=(?:\s*\btemp\b|\bOR\b|\bClassification\b))"
protein_pattern = r"protein value is (no|yes)"
browning_pattern = r"browning value is (\d+.\d)"
drying_pattern = r"drying value is (\d+.\d)"


def graph_to_json(graph, color_map_dict=None):
    """Convert a networkx graph to a JSON-serializable dict for vis.js"""
    nodes = []
    edges = []

    for node in graph.nodes():
        node_data = {"id": node, "label": node}
        if color_map_dict:
            code = color_map_dict.get(node, 0)
            if code == 1:
                node_data["color"] = "#ef4444"   # red - only in query
            elif code == 2:
                node_data["color"] = "#3b82f6"   # blue - only in candidate
            elif code == 3:
                node_data["color"] = "#a855f7"   # purple - overlap
            else:
                node_data["color"] = "#6b7280"   # gray
        nodes.append(node_data)

    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        edge = {"id": i, "from": u, "to": v}
        if "relationship" in data:
            edge["label"] = data["relationship"]
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def generate_recipe(ingredients: list[str]) -> str:
    model_name = "flax-community/t5-recipe-generation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_text = "items: " + ", ".join(ingredients)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    generated_output = model.generate(input_ids, max_new_tokens=512, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
    return tokenizer.decode(generated_output[0], skip_special_tokens=True)


class T5Classifier(nn.Module):
    def __init__(self, device, model_path):
        super(T5Classifier, self).__init__()
        self.t5 = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = None
        self.device = device

    def generate_for_batch(self, inputs):
        outputs = self.t5.generate(
            input_ids=inputs['input_ids'].squeeze(1).to(self.device),
            attention_mask=inputs['attention_mask'].squeeze(1).to(self.device),
            max_length=512,
        )
        outputs_decoded = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        input_decoded = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs['input_ids'].squeeze(1)]
        return {'inputs': input_decoded, 'outputs': outputs_decoded}

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.t5(input_ids, attention_mask, decoder_input_ids=decoder_input_ids, labels=labels, decoder_attention_mask=decoder_attention_mask)


class RecipeInference:
    def __init__(self, model_weights_path, model_path=None, max_sequence_length=512):
        self.MAX_SEQUENCE_LENGTH = max_sequence_length
        self.MODEL_PATH = model_path if model_path else self._detect_model_size(model_weights_path)
        print(f"Using model: {self.MODEL_PATH}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.MODEL_PATH, max_length=self.MAX_SEQUENCE_LENGTH, padding="max_length", truncation=True)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self._load_model(model_weights_path)

    def _detect_model_size(self, weights_path):
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            if 't5.shared.weight' in state_dict:
                embedding_dim = state_dict['t5.shared.weight'].shape[1]
            elif 't5.encoder.embed_tokens.weight' in state_dict:
                embedding_dim = state_dict['t5.encoder.embed_tokens.weight'].shape[1]
            else:
                for key in state_dict.keys():
                    if 'SelfAttention.q.weight' in key:
                        embedding_dim = state_dict[key].shape[0]
                        break
                else:
                    raise ValueError("Cannot determine model size")
            sizes = {512: 'google/mt5-small', 768: 'google/mt5-base', 1024: 'google/mt5-large', 2048: 'google/mt5-xl', 4096: 'google/mt5-xxl'}
            return sizes.get(embedding_dim, 'google/mt5-base')
        except Exception as e:
            print(f"Could not auto-detect model size: {e}. Defaulting to mt5-base")
            return 'google/mt5-base'

    def _load_model(self, weights_path):
        model = nn.DataParallel(T5Classifier(self.device, self.MODEL_PATH))
        model.module.tokenizer = self.tokenizer
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            model.module.load_state_dict(state_dict, strict=True)
            print("Model weights loaded successfully!")
        except RuntimeError as e:
            print(f"Trying strict=False... {e}")
            model.module.load_state_dict(state_dict, strict=False)
        if self.use_cuda:
            model = model.cuda()
        model.eval()
        return model

    def predict_from_text(self, recipe_text):
        input_text = f"Does the following recipe have oven settings? If yes, extract the oven settings. Recipe: {recipe_text}"
        inputs = self.tokenizer(input_text, padding='max_length', max_length=self.MAX_SEQUENCE_LENGTH, truncation=True, return_tensors="pt")
        if len(inputs['input_ids'].shape) == 1:
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        with torch.no_grad():
            results = self.model.module.generate_for_batch(inputs)
        return results['outputs'][0] if results['outputs'] else ""


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class RecipeProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_all()

    def _load_all(self):
        print("Loading knowledge graph and data files...")
        self.origin = nx.read_graphml("KG.graphml")
        self.recipes_df = pd.read_csv("prepared_recipes.csv")
        self.node_labels = nx.get_node_attributes(self.origin, 'type')

        
        print("Loading food data...")
        food_df = pd.read_csv('food.csv')
        food_df.columns = food_df.columns.str.strip()
        self.food_entities = [
            food.strip()
            for food in set(food_df['Category'].dropna().str.lower())
        ]
        # Pre-compile a single combined regex for fast ingredient matching
        escaped = [re.escape(f) for f in self.food_entities]
        self.food_pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE
        )

        print("Loading sentence transformer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.sentence_model = (
            AutoModel
            .from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            .to(self.device)
        )
        self.sentence_model.eval()

        print("Loading vocab and embeddings...")
        with open("node_type_vocab_mean.json", "r") as f:
            self.node_type_vocab = json.load(f)
        with open("edge_relationship_vocab_mean.json", "r") as f:
            self.edge_relationship_vocab = json.load(f)
        with open("subgraph_embeddings_mean.json", "r") as f:
            self.embeddings = json.load(f)

        print("Loading GCN model...")
        self.gcn_model = GCNEncoder(in_dim=405, hidden_dim=64, out_dim=128).to(self.device)
        self.gcn_model.load_state_dict(
            torch.load("gcn_model_mean.pth", map_location=self.device)
        )
        self.gcn_model.eval()

        print("Loading T5 settings extraction model...")
        self.recipe_inference = RecipeInference(
            "./t5_joint_settings_2023_05_15-09_02_53_AM.pt"
        )

        print("Loading T5 recipe generation model...")
        _gen_model_name = "flax-community/t5-recipe-generation"
        self.gen_tokenizer = AutoTokenizer.from_pretrained(_gen_model_name)
        self.gen_model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(_gen_model_name)
            .to(self.device)
        )
        self.gen_model.eval()

        print("Pre-computing food similarity inputs...")
        types = [v for u, v in self.origin.edges("Central")]
        self.foods = []
        for type_aux in types:
            self.foods += [v for u, v in self.origin.edges(type_aux)]

        self.food_similarity_inputs = {
            food: self._create_similarity_input(self.origin, food, self.embeddings)
            for food in self.foods
        }

        print("All models loaded successfully!")

    def analyze_recipe(self, instructions):
        try:
            return self.process_recipe(instructions)
        except Exception as e:
            return {"error": f"Error analyzing recipe: {str(e)}"}

    def process_recipe(self, instructions):
        extracted = self.recipe_inference.predict_from_text(instructions)
        kg = self._create_knowledge_graph(extracted, instructions)
        embeddings_dict = self._embed_subgraph(kg)[0]

        for key in embeddings_dict:
            embeddings_dict[key] = embeddings_dict[key][1]

        recipe_similarity_input = self._create_similarity_input(
            kg, "Query Recipe", embeddings_dict
        )

        similarity_score = -1000
        top_candidate = ""
        for food in self.foods:
            aux = self._compute_similarity(
                food, "Query Recipe",
                self.food_similarity_inputs[food],
                recipe_similarity_input, kg, embeddings_dict
            ).item()
            if similarity_score < aux:
                similarity_score = aux
                top_candidate = food

        sub_nodes = nx.descendants(self.origin, top_candidate)
        sub_nodes.add(top_candidate)
        candidate_subgraph = self.origin.subgraph(sub_nodes).copy()

        overlap_graph, colors_dict = self._create_common_graph(
            kg, candidate_subgraph, top_candidate
        )
        recipe = self._prepare_data_for_recipe(top_candidate)

        common_cnt = sum(1 for v in colors_dict.values() if v == 3)
        total_cnt  = len(colors_dict)
        overlap_pct = round(common_cnt / total_cnt * 100, 2) if total_cnt > 0 else 0

        return {
            "top_candidate":   top_candidate,
            "recipe":          recipe,
            "query_graph":     graph_to_json(kg),
            "candidate_graph": graph_to_json(candidate_subgraph),
            "overlap_graph":   graph_to_json(overlap_graph, colors_dict),
            "overlap_stats":   {
                "common": common_cnt, "total": total_cnt, "percent": overlap_pct
            },
        }

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _compute_node_edge_embedding(self, node, graph, edge_dim=20):
        vec = torch.zeros(edge_dim, dtype=torch.float32)
        rel_ids = [
            self.edge_relationship_vocab[d.get('relationship')]
            for _, _, d in graph.out_edges(node, data=True)
            if d.get('relationship') in self.edge_relationship_vocab
        ]
        for i in range(min(len(rel_ids), edge_dim)):
            vec[i] = rel_ids[i]
        return vec

    def _get_node_feature(self, node_name):
        name_processed = (
            node_name.split(' - ')[0]
            if "step" in node_name and " - " in node_name
            else node_name
        )
        encoded_input = self.tokenizer(
            name_processed, padding=True, truncation=True, return_tensors='pt'
        )
        # Move inputs to the same device as the model
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.sentence_model(**encoded_input)
        aux_embeddings = F.normalize(
            self._mean_pooling(model_output, encoded_input['attention_mask']),
            p=2, dim=1
        ).squeeze(0)
        multiplier = 1e6
        aux_embeddings = torch.round(aux_embeddings * multiplier) / multiplier
        label    = self.node_labels.get(node_name, '<unk>')
        label_id = torch.tensor(
            [self.node_type_vocab.get(label, self.node_type_vocab['<unk>'])],
            dtype=torch.float32, device=self.device
        )
        edge_embedding = self._compute_node_edge_embedding(node_name, self.origin).to(self.device)
        feature = torch.cat([aux_embeddings, label_id, edge_embedding], dim=0)
        return torch.round(feature * multiplier) / multiplier

    def _embed_subgraph(self, kg):
        sub_nodes = nx.descendants(kg, "Query Recipe")
        sub_nodes.add("Query Recipe")
        subgraph = kg.subgraph(sub_nodes).copy()
        features = []
        for node in subgraph.nodes():
            subgraph.nodes[node]['x'] = self._get_node_feature(node)
            features.append(subgraph.nodes[node]['x'])
        data = from_networkx(subgraph)
        data.x          = torch.stack(features).to(self.device)
        data.edge_index = data.edge_index.to(self.device)
        with torch.no_grad():
            embeddings = self.gcn_model(data.x, data.edge_index)
        return (
            {node: (subgraph.nodes[node]['type'], embeddings[i].tolist())
             for i, node in enumerate(subgraph.nodes())},
            subgraph,
        )

    def _extract_baking_info(self, text):
        if not isinstance(text, str):
            return 0, [], [], []
        steps = re.findall(step_pattern, text, flags=re.IGNORECASE)
        temp_by_step, time_by_step, func_by_step = [], [], []
        for step in steps:
            temp_by_step.append(re.findall(temp_pattern, step, flags=re.IGNORECASE))
            time_by_step.append(re.findall(time_pattern, step, flags=re.IGNORECASE))
            func_by_step.append(re.findall(mode_pattern, step, flags=re.IGNORECASE))
        return len(steps), temp_by_step, time_by_step, func_by_step

    def _create_knowledge_graph(self, extracted, instructions):
        kg = nx.DiGraph()
        kg.add_node("Query Recipe", type="title")
        _, temps, times, modes = self._extract_baking_info(extracted)
        temps.reverse(); times.reverse(); modes.reverse()

        for temp in temps:
            if temp[0] not in kg.nodes():
                kg.add_node(temp[0].strip(), type="temperature")
        for time in times:
            if time[0] not in kg.nodes():
                kg.add_node(time[0].strip(), type="time")
        for mode in modes:
            if mode[0] not in kg.nodes():
                kg.add_node(mode[0].strip(), type="mode")

        no_steps = len(temps)
        cnt = no_steps - 1
        for aux in range(len(temps)):
            kg.add_node(f"step{cnt}", type="step")
            kg.add_edge(f"step{cnt}", temps[aux][0].strip(), relationship="has temperature")
            kg.add_edge(f"step{cnt}", times[aux][0].strip(), relationship="has time")
            kg.add_edge(f"step{cnt}", modes[aux][0].strip(), relationship="has mode")
            cnt -= 1

        cnt = no_steps
        for _ in range(len(temps) - 1):
            kg.add_edge(f"step{cnt - 1}", f"step{cnt - 2}", relationship="based on")
            cnt -= 1
        if no_steps > 0:
            kg.add_edge("Query Recipe", f"step{no_steps - 1}", relationship="based on")

        words = re.findall(r'\b\w+\b', str(instructions).lower())
        try:
            joined      = " ||| ".join(words)
            translated  = GoogleTranslator(source='auto', target='en').translate(joined)
            trans_words = [w.strip().lower() for w in translated.split("|||")]
            if len(trans_words) != len(words):
                trans_words = words
        except Exception:
            trans_words = words
        instructions_translated = " ".join(trans_words)

        for match in self.food_pattern.finditer(instructions_translated):
            food = match.group(0).lower()
            kg.add_node(food, type="ingredient")
            kg.add_edge("Query Recipe", food, relationship="contains")

        protein  = re.findall(protein_pattern,  extracted, flags=re.IGNORECASE)
        browning = re.findall(browning_pattern, extracted, flags=re.IGNORECASE)
        drying   = re.findall(drying_pattern,   extracted, flags=re.IGNORECASE)
        if protein and browning and drying:
            kg.add_node(f"Browning:{int(float(browning[0]))}", type="browning")
            kg.add_edge("Query Recipe", f"Browning:{int(float(browning[0]))}", relationship="has property")
            kg.add_node(f"Protein:{protein[0]}", type="protein")
            kg.add_edge("Query Recipe", f"Protein:{protein[0]}", relationship="has property")
            kg.add_node(f"Drying:{int(float(drying[0]))}", type="drying")
            kg.add_edge("Query Recipe", f"Drying:{int(float(drying[0]))}", relationship="has property")
        return kg

    def _create_similarity_input(self, kg, candidate, features):
        sub = nx.descendants(kg, candidate)
        candidate_nodes = {node: (kg.nodes[node]['type'], features[node]) for node in sub}
        result_dict = {"ingredient": []}
        for node, (node_type, feat) in candidate_nodes.items():
            if node_type == "browning":    result_dict["browning"]  = feat
            elif node_type == "drying":    result_dict["drying"]    = feat
            elif node_type == "protein":   result_dict["protein"]   = feat
            elif node_type == "ingredient":result_dict["ingredient"].append(feat)
            elif node_type == "step":
                result_dict[node] = [0, 0, 0]
                for child in kg.successors(node):
                    ct = kg.nodes[child].get("type")
                    if ct == "temperature": result_dict[node][0] = candidate_nodes[child][1]
                    elif ct == "mode":      result_dict[node][1] = candidate_nodes[child][1]
                    elif ct == "time":      result_dict[node][2] = candidate_nodes[child][1]
        return result_dict

    def _compute_similarity(self, candidate_node, original_node,
                            candidate_inputs, original_inputs, kg, embeddings_dict):
        candidate_nodes = {
            node: (self.origin.nodes[node]['type'], self.embeddings[node])
            for node in nx.descendants(self.origin, candidate_node)
        }
        original_nodes = {
            node: (kg.nodes[node]['type'], embeddings_dict[node])
            for node in nx.descendants(kg, original_node)
        }
        len_candidate = sum(1 for v in candidate_nodes.values() if v[0] == 'step')
        len_original  = sum(1 for v in original_nodes.values()  if v[0] == 'step')
        length_backbone = min(len_candidate, len_original)
        candidate_steps = sorted([
            k for k in candidate_nodes
            if any(f"step{i} - " in k for i in range(length_backbone))
        ])
        original_steps = sorted([
            k for k in original_nodes
            if any(f"step{i}" in k for i in range(length_backbone))
        ])
        no_candidate_nodes = len(candidate_nodes)
        no_common_nodes    = sum(1 for node in kg.nodes() if node in candidate_nodes)
        similarity = sum(
            self._cos_sim(candidate_nodes[cs][1], original_nodes[os][1]) +
            self._cos_sim(candidate_inputs[cs][0], original_inputs[os][0]) +
            self._cos_sim(candidate_inputs[cs][1], original_inputs[os][1]) +
            self._cos_sim(candidate_inputs[cs][2], original_inputs[os][2])
            for cs, os in zip(candidate_steps, original_steps)
        )
        similarity += (
            self._cos_sim(candidate_inputs.get('browning', [0]), original_inputs.get('browning', [0])) +
            self._cos_sim(candidate_inputs.get('drying',   [0]), original_inputs.get('drying',   [0])) +
            self._cos_sim(candidate_inputs.get('protein',  [0]), original_inputs.get('protein',  [0]))
        )
        no_candidate_ingredients = sum(
            1 for v in candidate_nodes.values() if v[0] == 'ingredient'
        )
        no_common_ingredients = sum(
            1 for node in kg.nodes()
            if node in candidate_nodes and candidate_nodes[node][0] == 'ingredient'
        )
        if no_candidate_ingredients > 0:
            aux_score, _ = self._optimal_pairing(
                candidate_inputs['ingredient'], original_inputs['ingredient']
            )
            similarity += aux_score * no_common_ingredients / no_candidate_ingredients
        denom = no_candidate_nodes - 1 - length_backbone
        return similarity * (no_common_nodes - length_backbone) / denom \
            if denom != 0 else torch.tensor(0.0)

    def _cos_sim(self, list1, list2):
        return cosine_similarity(
            torch.tensor(list1).unsqueeze(0),
            torch.tensor(list2).unsqueeze(0)
        )

    def _optimal_pairing(self, original_ingredients_list, candidate_ingredients_list):
        m, n = len(original_ingredients_list), len(candidate_ingredients_list)
        if m == 0 or n == 0:
            return 0.0, []
        orig_array = np.array(original_ingredients_list)
        cand_array = np.array(candidate_ingredients_list)
        orig_norm = np.where(
            np.linalg.norm(orig_array, axis=1, keepdims=True) == 0, 1,
            np.linalg.norm(orig_array, axis=1, keepdims=True)
        )
        cand_norm = np.where(
            np.linalg.norm(cand_array, axis=1, keepdims=True) == 0, 1,
            np.linalg.norm(cand_array, axis=1, keepdims=True)
        )
        similarity_matrix = np.dot(orig_array / orig_norm, (cand_array / cand_norm).T)
        indexes = Munkres().compute((-similarity_matrix).tolist())
        row_indices, col_indices = zip(*indexes)
        return float(similarity_matrix[row_indices, col_indices].sum()), list(zip(row_indices, col_indices))

    def _create_common_graph(self, kg, origin_subgraph, candidate_title):
        graph = nx.compose(kg, origin_subgraph)
        graph.add_node("Overlap Graph", type="title")
        query_neighbors     = list(kg.successors("Query Recipe"))         if "Query Recipe"   in kg             else []
        candidate_neighbors = list(origin_subgraph.successors(candidate_title)) if candidate_title in origin_subgraph else []
        for node in set(query_neighbors + candidate_neighbors):
            graph.add_edge("Overlap Graph", node)
        colors_dict = {}
        for node in kg.nodes():
            if node == "Query Recipe": continue
            colors_dict[node] = 3 if node in origin_subgraph.nodes() else 1
        for node in origin_subgraph.nodes():
            if node != candidate_title and node not in colors_dict:
                colors_dict[node] = 2
        colors_dict["Overlap Graph"] = 3
        if candidate_title in graph: graph.remove_node(candidate_title)
        if "Query Recipe"  in graph: graph.remove_node("Query Recipe")
        return graph, colors_dict

    def _generate_recipe(self, ingredients: list) -> str:
        input_text = "items: " + ", ".join(ingredients)
        input_ids = self.gen_tokenizer(
            input_text, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            generated_output = self.gen_model.generate(
                input_ids,
                max_new_tokens=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        return self.gen_tokenizer.decode(generated_output[0], skip_special_tokens=True)

    def _prepare_data_for_recipe(self, recipe_title):
        sub_nodes = nx.descendants(self.origin, recipe_title)
        sub_nodes.add(recipe_title)
        subgraph = self.origin.subgraph(sub_nodes).copy()
        ingredients, steps = [], {}
        for edge in subgraph.edges():
            rel = subgraph.edges[edge]['relationship']
            if rel == 'contains':
                ingredients.append(edge[1])
            elif rel == 'based on':
                steps[edge[1].split(' - ')[0]] = ["", "", ""]
        for edge in subgraph.edges():
            rel = subgraph.edges[edge]['relationship']
            key = edge[0].split(' - ')[0]
            if rel == 'has temperature': steps[key][0] = edge[1]
            elif rel == 'has time':      steps[key][1] = edge[1]
            elif rel == 'has mode':      steps[key][2] = edge[1]
        directions = "directions from the graph:"
        for i in range(len(steps)):
            key = f"step{i}"
            if key in steps:
                directions += f" Step {i+1}: bake at {steps[key][0]} for {steps[key][1]} using the mode: {steps[key][2]}."
            else:
                break
        aux   = self._generate_recipe(ingredients)
        start = aux.find("title:")
        end   = aux.find("ingredients:", start)
        if start != -1 and end != -1:
            aux = "title: " + recipe_title + "\n" + aux[end:]
        start = aux.find("directions:")
        if start != -1:
            aux = aux[:start] + "\ndirections suggested by the LLM:" + aux[start + len("directions:"):]
        return aux + "\n" + directions
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_all()

    def _load_all(self):
        print("Loading knowledge graph and data files...")
        self.origin = nx.read_graphml("KG.graphml")
        self.recipes_df = pd.read_csv("prepared_recipes.csv")
        self.node_labels = nx.get_node_attributes(self.origin, 'type')

        print("Loading food data...")
        food_df = pd.read_csv('food.csv')
        food_df.columns = food_df.columns.str.strip()
        self.food_entities = [
            food.strip()
            for food in set(food_df['Category'].dropna().str.lower())
        ]
        # Pre-compile a single combined regex for fast ingredient matching
        escaped = [re.escape(f) for f in self.food_entities]
        self.food_pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE
        )

        print("Loading sentence transformer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.sentence_model = (
            AutoModel
            .from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            .to(self.device)
        )
        self.sentence_model.eval()

        print("Loading vocab and embeddings...")
        with open("node_type_vocab_mean.json", "r") as f:
            self.node_type_vocab = json.load(f)
        with open("edge_relationship_vocab_mean.json", "r") as f:
            self.edge_relationship_vocab = json.load(f)
        with open("subgraph_embeddings_mean.json", "r") as f:
            self.embeddings = json.load(f)

        print("Loading GCN model...")
        self.gcn_model = GCNEncoder(in_dim=405, hidden_dim=64, out_dim=128).to(self.device)
        self.gcn_model.load_state_dict(
            torch.load("gcn_model_mean.pth", map_location=self.device)
        )
        self.gcn_model.eval()

        print("Loading T5 model...")
        self.recipe_inference = RecipeInference(
            "./t5_joint_settings_2023_05_15-09_02_53_AM.pt"
        )

        print("Pre-computing food similarity inputs...")
        types = [v for u, v in self.origin.edges("Central")]
        self.foods = []
        for type_aux in types:
            self.foods += [v for u, v in self.origin.edges(type_aux)]

        self.food_similarity_inputs = {
            food: self._create_similarity_input(self.origin, food, self.embeddings)
            for food in self.foods
        }

        print("All models loaded successfully!")

    def analyze_recipe(self, instructions):
        try:
            return self.process_recipe(instructions)
        except Exception as e:
            return {"error": f"Error analyzing recipe: {str(e)}"}

    def process_recipe(self, instructions):
        extracted = self.recipe_inference.predict_from_text(instructions)
        kg = self._create_knowledge_graph(extracted, instructions)
        embeddings_dict = self._embed_subgraph(kg)[0]

        for key in embeddings_dict:
            embeddings_dict[key] = embeddings_dict[key][1]

        recipe_similarity_input = self._create_similarity_input(
            kg, "Query Recipe", embeddings_dict
        )

        similarity_score = -1000
        top_candidate = ""
        for food in self.foods:
            aux = self._compute_similarity(
                food, "Query Recipe",
                self.food_similarity_inputs[food],
                recipe_similarity_input, kg, embeddings_dict
            ).item()
            if similarity_score < aux:
                similarity_score = aux
                top_candidate = food

        sub_nodes = nx.descendants(self.origin, top_candidate)
        sub_nodes.add(top_candidate)
        candidate_subgraph = self.origin.subgraph(sub_nodes).copy()

        overlap_graph, colors_dict = self._create_common_graph(
            kg, candidate_subgraph, top_candidate
        )
        recipe = self._prepare_data_for_recipe(top_candidate)

        common_cnt = sum(1 for v in colors_dict.values() if v == 3)
        total_cnt  = len(colors_dict)
        overlap_pct = round(common_cnt / total_cnt * 100, 2) if total_cnt > 0 else 0

        return {
            "top_candidate":   top_candidate,
            "recipe":          recipe,
            "query_graph":     graph_to_json(kg),
            "candidate_graph": graph_to_json(candidate_subgraph),
            "overlap_graph":   graph_to_json(overlap_graph, colors_dict),
            "overlap_stats":   {
                "common": common_cnt, "total": total_cnt, "percent": overlap_pct
            },
        }

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _compute_node_edge_embedding(self, node, graph, edge_dim=20):
        vec = torch.zeros(edge_dim, dtype=torch.float32)
        rel_ids = [
            self.edge_relationship_vocab[d.get('relationship')]
            for _, _, d in graph.out_edges(node, data=True)
            if d.get('relationship') in self.edge_relationship_vocab
        ]
        for i in range(min(len(rel_ids), edge_dim)):
            vec[i] = rel_ids[i]
        return vec

    def _get_node_feature(self, node_name):
        name_processed = (
            node_name.split(' - ')[0]
            if "step" in node_name and " - " in node_name
            else node_name
        )
        encoded_input = self.tokenizer(
            name_processed, padding=True, truncation=True, return_tensors='pt'
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.sentence_model(**encoded_input)
        aux_embeddings = F.normalize(
            self._mean_pooling(model_output, encoded_input['attention_mask']),
            p=2, dim=1
        ).squeeze(0)
        multiplier = 1e6
        aux_embeddings = torch.round(aux_embeddings * multiplier) / multiplier
        label    = self.node_labels.get(node_name, '<unk>')
        label_id = torch.tensor(
            [self.node_type_vocab.get(label, self.node_type_vocab['<unk>'])],
            dtype=torch.float32, device=self.device
        )
        edge_embedding = self._compute_node_edge_embedding(node_name, self.origin).to(self.device)
        feature = torch.cat([aux_embeddings, label_id, edge_embedding], dim=0)
        return torch.round(feature * multiplier) / multiplier

    def _embed_subgraph(self, kg):
        sub_nodes = nx.descendants(kg, "Query Recipe")
        sub_nodes.add("Query Recipe")
        subgraph = kg.subgraph(sub_nodes).copy()
        features = []
        for node in subgraph.nodes():
            subgraph.nodes[node]['x'] = self._get_node_feature(node)
            features.append(subgraph.nodes[node]['x'])
        data = from_networkx(subgraph)
        data.x          = torch.stack(features).to(self.device)
        data.edge_index = data.edge_index.to(self.device)
        with torch.no_grad():
            embeddings = self.gcn_model(data.x, data.edge_index)
        return (
            {node: (subgraph.nodes[node]['type'], embeddings[i].tolist())
             for i, node in enumerate(subgraph.nodes())},
            subgraph,
        )

    def _extract_baking_info(self, text):
        if not isinstance(text, str):
            return 0, [], [], []
        steps = re.findall(step_pattern, text, flags=re.IGNORECASE)
        temp_by_step, time_by_step, func_by_step = [], [], []
        for step in steps:
            temp_by_step.append(re.findall(temp_pattern, step, flags=re.IGNORECASE))
            time_by_step.append(re.findall(time_pattern, step, flags=re.IGNORECASE))
            func_by_step.append(re.findall(mode_pattern, step, flags=re.IGNORECASE))
        return len(steps), temp_by_step, time_by_step, func_by_step

    def _create_knowledge_graph(self, extracted, instructions):
        kg = nx.DiGraph()
        kg.add_node("Query Recipe", type="title")
        _, temps, times, modes = self._extract_baking_info(extracted)
        temps.reverse(); times.reverse(); modes.reverse()

        for temp in temps:
            if temp[0] not in kg.nodes():
                kg.add_node(temp[0].strip(), type="temperature")
        for time in times:
            if time[0] not in kg.nodes():
                kg.add_node(time[0].strip(), type="time")
        for mode in modes:
            if mode[0] not in kg.nodes():
                kg.add_node(mode[0].strip(), type="mode")

        no_steps = len(temps)
        cnt = no_steps - 1
        for aux in range(len(temps)):
            kg.add_node(f"step{cnt}", type="step")
            kg.add_edge(f"step{cnt}", temps[aux][0].strip(), relationship="has temperature")
            kg.add_edge(f"step{cnt}", times[aux][0].strip(), relationship="has time")
            kg.add_edge(f"step{cnt}", modes[aux][0].strip(), relationship="has mode")
            cnt -= 1

        cnt = no_steps
        for _ in range(len(temps) - 1):
            kg.add_edge(f"step{cnt - 1}", f"step{cnt - 2}", relationship="based on")
            cnt -= 1
        if no_steps > 0:
            kg.add_edge("Query Recipe", f"step{no_steps - 1}", relationship="based on")

        words = re.findall(r'\b\w+\b', str(instructions).lower())
        try:
            joined      = " ||| ".join(words)
            translated  = GoogleTranslator(source='auto', target='en').translate(joined)
            trans_words = [w.strip().lower() for w in translated.split("|||")]
            if len(trans_words) != len(words):
                trans_words = words
        except Exception:
            trans_words = words
        instructions_translated = " ".join(trans_words)

        for match in self.food_pattern.finditer(instructions_translated):
            food = match.group(0).lower()
            kg.add_node(food, type="ingredient")
            kg.add_edge("Query Recipe", food, relationship="contains")

        protein  = re.findall(protein_pattern,  extracted, flags=re.IGNORECASE)
        browning = re.findall(browning_pattern, extracted, flags=re.IGNORECASE)
        drying   = re.findall(drying_pattern,   extracted, flags=re.IGNORECASE)
        if protein and browning and drying:
            kg.add_node(f"Browning:{int(float(browning[0]))}", type="browning")
            kg.add_edge("Query Recipe", f"Browning:{int(float(browning[0]))}", relationship="has property")
            kg.add_node(f"Protein:{protein[0]}", type="protein")
            kg.add_edge("Query Recipe", f"Protein:{protein[0]}", relationship="has property")
            kg.add_node(f"Drying:{int(float(drying[0]))}", type="drying")
            kg.add_edge("Query Recipe", f"Drying:{int(float(drying[0]))}", relationship="has property")
        return kg

    def _create_similarity_input(self, kg, candidate, features):
        sub = nx.descendants(kg, candidate)
        candidate_nodes = {node: (kg.nodes[node]['type'], features[node]) for node in sub}
        result_dict = {"ingredient": []}
        for node, (node_type, feat) in candidate_nodes.items():
            if node_type == "browning":    result_dict["browning"]  = feat
            elif node_type == "drying":    result_dict["drying"]    = feat
            elif node_type == "protein":   result_dict["protein"]   = feat
            elif node_type == "ingredient":result_dict["ingredient"].append(feat)
            elif node_type == "step":
                result_dict[node] = [0, 0, 0]
                for child in kg.successors(node):
                    ct = kg.nodes[child].get("type")
                    if ct == "temperature": result_dict[node][0] = candidate_nodes[child][1]
                    elif ct == "mode":      result_dict[node][1] = candidate_nodes[child][1]
                    elif ct == "time":      result_dict[node][2] = candidate_nodes[child][1]
        return result_dict

    def _compute_similarity(self, candidate_node, original_node,
                            candidate_inputs, original_inputs, kg, embeddings_dict):
        candidate_nodes = {
            node: (self.origin.nodes[node]['type'], self.embeddings[node])
            for node in nx.descendants(self.origin, candidate_node)
        }
        original_nodes = {
            node: (kg.nodes[node]['type'], embeddings_dict[node])
            for node in nx.descendants(kg, original_node)
        }
        len_candidate = sum(1 for v in candidate_nodes.values() if v[0] == 'step')
        len_original  = sum(1 for v in original_nodes.values()  if v[0] == 'step')
        length_backbone = min(len_candidate, len_original)
        candidate_steps = sorted([
            k for k in candidate_nodes
            if any(f"step{i} - " in k for i in range(length_backbone))
        ])
        original_steps = sorted([
            k for k in original_nodes
            if any(f"step{i}" in k for i in range(length_backbone))
        ])
        no_candidate_nodes = len(candidate_nodes)
        no_common_nodes    = sum(1 for node in kg.nodes() if node in candidate_nodes)
        similarity = sum(
            self._cos_sim(candidate_nodes[cs][1], original_nodes[os][1]) +
            self._cos_sim(candidate_inputs[cs][0], original_inputs[os][0]) +
            self._cos_sim(candidate_inputs[cs][1], original_inputs[os][1]) +
            self._cos_sim(candidate_inputs[cs][2], original_inputs[os][2])
            for cs, os in zip(candidate_steps, original_steps)
        )
        similarity += (
            self._cos_sim(candidate_inputs.get('browning', [0]), original_inputs.get('browning', [0])) +
            self._cos_sim(candidate_inputs.get('drying',   [0]), original_inputs.get('drying',   [0])) +
            self._cos_sim(candidate_inputs.get('protein',  [0]), original_inputs.get('protein',  [0]))
        )
        no_candidate_ingredients = sum(
            1 for v in candidate_nodes.values() if v[0] == 'ingredient'
        )
        no_common_ingredients = sum(
            1 for node in kg.nodes()
            if node in candidate_nodes and candidate_nodes[node][0] == 'ingredient'
        )
        if no_candidate_ingredients > 0:
            aux_score, _ = self._optimal_pairing(
                candidate_inputs['ingredient'], original_inputs['ingredient']
            )
            similarity += aux_score * no_common_ingredients / no_candidate_ingredients
        denom = no_candidate_nodes - 1 - length_backbone
        return similarity * (no_common_nodes - length_backbone) / denom \
            if denom != 0 else torch.tensor(0.0)

    def _cos_sim(self, list1, list2):
        return cosine_similarity(
            torch.tensor(list1).unsqueeze(0),
            torch.tensor(list2).unsqueeze(0)
        )

    def _optimal_pairing(self, original_ingredients_list, candidate_ingredients_list):
        m, n = len(original_ingredients_list), len(candidate_ingredients_list)
        if m == 0 or n == 0:
            return 0.0, []
        orig_array = np.array(original_ingredients_list)
        cand_array = np.array(candidate_ingredients_list)
        orig_norm = np.where(
            np.linalg.norm(orig_array, axis=1, keepdims=True) == 0, 1,
            np.linalg.norm(orig_array, axis=1, keepdims=True)
        )
        cand_norm = np.where(
            np.linalg.norm(cand_array, axis=1, keepdims=True) == 0, 1,
            np.linalg.norm(cand_array, axis=1, keepdims=True)
        )
        similarity_matrix = np.dot(orig_array / orig_norm, (cand_array / cand_norm).T)
        indexes = Munkres().compute((-similarity_matrix).tolist())
        row_indices, col_indices = zip(*indexes)
        return float(similarity_matrix[row_indices, col_indices].sum()), list(zip(row_indices, col_indices))

    def _create_common_graph(self, kg, origin_subgraph, candidate_title):
        graph = nx.compose(kg, origin_subgraph)
        graph.add_node("Overlap Graph", type="title")
        query_neighbors     = list(kg.successors("Query Recipe"))         if "Query Recipe"   in kg             else []
        candidate_neighbors = list(origin_subgraph.successors(candidate_title)) if candidate_title in origin_subgraph else []
        for node in set(query_neighbors + candidate_neighbors):
            graph.add_edge("Overlap Graph", node)
        colors_dict = {}
        for node in kg.nodes():
            if node == "Query Recipe": continue
            colors_dict[node] = 3 if node in origin_subgraph.nodes() else 1
        for node in origin_subgraph.nodes():
            if node != candidate_title and node not in colors_dict:
                colors_dict[node] = 2
        colors_dict["Overlap Graph"] = 3
        if candidate_title in graph: graph.remove_node(candidate_title)
        if "Query Recipe"  in graph: graph.remove_node("Query Recipe")
        return graph, colors_dict

    def _prepare_data_for_recipe(self, recipe_title):
        sub_nodes = nx.descendants(self.origin, recipe_title)
        sub_nodes.add(recipe_title)
        subgraph = self.origin.subgraph(sub_nodes).copy()
        ingredients, steps = [], {}
        for edge in subgraph.edges():
            rel = subgraph.edges[edge]['relationship']
            if rel == 'contains':
                ingredients.append(edge[1])
            elif rel == 'based on':
                steps[edge[1].split(' - ')[0]] = ["", "", ""]
        for edge in subgraph.edges():
            rel = subgraph.edges[edge]['relationship']
            key = edge[0].split(' - ')[0]
            if rel == 'has temperature': steps[key][0] = edge[1]
            elif rel == 'has time':      steps[key][1] = edge[1]
            elif rel == 'has mode':      steps[key][2] = edge[1]
        directions = "directions from the graph:"
        for i in range(len(steps)):
            key = f"step{i}"
            if key in steps:
                directions += f" Step {i+1}: bake at {steps[key][0]} for {steps[key][1]} using the mode: {steps[key][2]}."
            else:
                break
        aux   = generate_recipe(ingredients)
        start = aux.find("title:")
        end   = aux.find("ingredients:", start)
        if start != -1 and end != -1:
            aux = "title: " + recipe_title + "\n" + aux[end:]
        start = aux.find("directions:")
        if start != -1:
            aux = aux[:start] + "\ndirections suggested by the LLM:" + aux[start + len("directions:"):]
        return aux + "\n" + directions