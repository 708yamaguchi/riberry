#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import json
from std_msgs.msg import String
from speech_recognition_msgs.msg import SpeechRecognitionCandidates

from riberry.task_parser import TaskParser
from riberry_startup.srv import TaskInstruction, TaskInstructionRequest

class TaskParserNode:
    def __init__(self):
        rospy.init_node("task_parser_node", anonymous=True)

        config_path = rospy.get_param("~task_config_path")
        rospy.loginfo(f"Loading task config from: {config_path}")

        # 1回目の読み込み（ログ表示用とバリデーション用）
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config file: {e}")

        # 2回目の読み込み（Parser用）
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load config file: {e}")
            exit(1)

        # Parserの初期化
        # 初期化中にEmbeddingの計算やWarmupが走るため少し時間がかかります
        self.parser = TaskParser(config=config_dict)

        # ROS通信の設定
        self.srv_client = rospy.ServiceProxy('/task_instruction', TaskInstruction)

        # 入力トピック: 音声認識結果
        self.sub_speech = rospy.Subscriber(
            "speech_to_text",
            SpeechRecognitionCandidates,
            self._cb_speech
        )
        # 出力トピック: 解析結果のJSON文字列
        self.pub_command = rospy.Publisher(
            "/task_command_json",
            String,
            queue_size=1
        )

        rospy.loginfo("Task Parser Node is ready.")

    def _cb_speech(self, msg):
        """
        音声認識結果(Candidates)を受け取り、最も確度が高い transcript[0] を解析する
        """
        if not msg.transcript:
            rospy.logwarn("Received empty transcript.")
            return

        # 一番信頼度の高い候補を採用
        input_text = msg.transcript[0]
        rospy.loginfo(f"Received speech: '{input_text}'")

        # 解析実行
        # 関係ない発話の場合は 空の辞書 {} または None が返る前提
        result_json = self.parser.parse(input_text)

        # 空の場合は「関係ない発話」とみなして処理をスキップ
        if not result_json:
            rospy.loginfo("Irrelevant speech detected (Parsed result is empty). Skipping execution.")
            return

        # 有効なタスク指示がある場合のみ以下を実行
        json_str = json.dumps(result_json, ensure_ascii=False)
        rospy.loginfo(f"Parsed result: {json_str}")
        self.pub_command.publish(json_str)
        self._call_service(result_json)

    def _call_service(self, data):
        """解析結果をROSサービスのリクエストに変換して送信"""
        try:
            # サービスが有効になるまで少し待つ
            rospy.wait_for_service('/task_instruction', timeout=1.0)

            req = TaskInstructionRequest()
            req.target_object = data.get("Target")
            req.action_verb   = data.get("Action")

            repeat = data.get("Repeat", {})
            req.repeat_value  = int(repeat.get("Value", 1))
            req.repeat_unit   = repeat.get("Unit", "times")

            resp = self.srv_client(req)

            if resp.success:
                rospy.loginfo("Service Call Success: Task accepted.")
            else:
                rospy.logwarn(f"Service Call Rejected: {resp.message}")

        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"Service call failed: {e}")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = TaskParserNode()
    node.run()
