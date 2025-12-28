import ollama
import json
import numpy as np

class TaskParser:
    def __init__(self, config, chat_model='qwen2.5', embed_model='qwen2.5'):
        """
        Args:
            config (dict): タスク設定辞書 (action_registry, object_registry等を含む)
            chat_model (str): チャット用LLMモデル名
            embed_model (str): Embedding用モデル名
        """
        self.config = config
        self.chat_model = chat_model
        self.embed_model = embed_model
        
        # 1. 候補リストの展開
        self.valid_actions = list(self.config.get("action_registry", {}).keys())
        # _default は除外し、純粋なターゲットのみリスト化
        self.valid_objects = [k for k in self.config.get("object_registry", {}).keys() if k != "_default"]
        self.valid_units = self.config.get("valid_units", ["times", "minutes"])

        print("[TaskParser] Initializing: Pre-calculating embeddings...")
        
        # 2. Embedding事前計算
        self.action_vectors = self._precompute_vectors(self.valid_actions)
        self.object_vectors = self._precompute_vectors(self.valid_objects)
        self.unit_vectors   = self._precompute_vectors(self.valid_units)
        
        # 3. ウォームアップ (Prompt Cachingのため)
        print("[TaskParser] Warming up Chat Model...")
        self._warm_up()
        print("[TaskParser] Ready.")

    def _get_embedding(self, text):
        try:
            response = ollama.embeddings(model=self.embed_model, prompt=text)
            return np.array(response["embedding"])
        except Exception as e:
            print(f"Embedding Error: {e}")
            return np.zeros(1)

    def _precompute_vectors(self, candidates):
        cache = {}
        for word in candidates:
            cache[word] = self._get_embedding(word)
        return cache

    def _build_system_prompt(self):
        return f"""
        あなたはロボットへの命令を解析するアシスタントです。
        ユーザーの入力文から情報を抽出し、必ず以下の制約に従ってJSON形式で出力してください。

        【制約事項: 候補リストからの選択】
        抽出する値は、必ず以下のリストの中から最も意味が近いものを選んでください。
        リストにない言葉は使わないでください。
        
        - Target (対象): {self.valid_objects}
        - Action (動作): {self.valid_actions}
        - Repeat.Unit (単位): {self.valid_units} (回数ならtimes, 時間ならminutes)

        【出力フォーマット】
        {{
            "Target": "...", 
            "Action": "...",
            "Repeat": {{
                "Value": 数値 (数値がない場合は 1),
                "Unit": "..."
            }}
        }}
        
        入力文の意味を解釈し、"洗って" -> "clean" のように適切な変換を行ってください。
        """

    def _warm_up(self):
        system_prompt = self._build_system_prompt()
        try:
            ollama.chat(
                model=self.chat_model,
                format='json',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'warmup'}
                ]
            )
        except Exception as e:
            print(f"Warmup warning: {e}")

    def _find_best_match_from_cache(self, query_text, candidate_cache):
        # 入力が空(Noneや"")の場合は None を返す
        if not query_text or not candidate_cache:
            return None

        if query_text in candidate_cache:
            return query_text

        query_vec = self._get_embedding(query_text)
        best_candidate = None
        best_score = -1.0

        for candidate, cand_vec in candidate_cache.items():
            norm_q = np.linalg.norm(query_vec)
            norm_c = np.linalg.norm(cand_vec)
            
            if norm_q == 0 or norm_c == 0:
                score = 0.0
            else:
                score = np.dot(query_vec, cand_vec) / (norm_q * norm_c)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate

    def parse(self, text):
        system_prompt = self._build_system_prompt()

        try:
            response = ollama.chat(
                model=self.chat_model,
                format='json',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': text},
                ]
            )
            content = response['message']['content']
            llm_result = json.loads(content)
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

        # --- Stage 2: Embedding Matching (Double Check) ---
        final_result = llm_result.copy()

        # Target (空文字ならNone)
        raw_target = llm_result.get("Target", "")
        final_result["Target"] = self._find_best_match_from_cache(raw_target, self.object_vectors)

        # Action (空文字ならNone)
        raw_action = llm_result.get("Action", "")
        final_result["Action"] = self._find_best_match_from_cache(raw_action, self.action_vectors)

        # Repeat情報の補正
        if "Repeat" in llm_result:
            if "Value" not in final_result["Repeat"] or final_result["Repeat"]["Value"] is None:
                 final_result["Repeat"]["Value"] = 1

            raw_unit = llm_result["Repeat"].get("Unit", "")
            matched_unit = self._find_best_match_from_cache(raw_unit, self.unit_vectors)

            # Unitが特定できない(None)場合は、言語的な省略とみなして "times" を埋める
            if matched_unit is None:
                final_result["Repeat"]["Unit"] = "times"
            else:
                final_result["Repeat"]["Unit"] = matched_unit
        else:
            final_result["Repeat"] = { "Value": 1, "Unit": "times" }

        return final_result


# ==========================================================
#  Standalone Test
# ==========================================================
if __name__ == "__main__":
    # ベタ書き設定
    dummy_config = {
        "action_registry": {
            "clean": {},
            "collect": {},
            "paint": {}
        },
        "object_registry": {
            "screws": {},
            "coffee powder": {},
            "dust": {}
        },
        "valid_units": ["times", "minutes"]
    }

    print("--- Starting Standalone Test ---")
    # 設定辞書を直接渡す
    parser = TaskParser(config=dummy_config)

    test_sentences = [
        "ネジを全部集めて。",
        "コーヒーの粉を3回掃除してください。",
        "5分間、塗ってください。"
    ]

    for text in test_sentences:
        print(f"\nInput: {text}")
        result = parser.parse(text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
