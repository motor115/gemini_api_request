import functions_framework
from flask import jsonify, request
import google.auth
import google.auth.transport.requests
import requests
import os
import json
import logging

# ロギングを設定
logging.basicConfig(level=logging.INFO)

@functions_framework.http
def gemini_text_api(request):
    """
    Vertex AI Gemini APIへのプロキシとして機能するCloud Function
    テキストからテキストを生成するAPIエンドポイント
    """
    # CORSヘッダーの設定
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    try:
        logging.info("リクエスト受信")
        
        # リクエストデータの取得
        request_json = request.get_json(silent=True)
        if not request_json:
            logging.error("リクエストデータなし")
            return jsonify({'error': 'リクエストデータが必要です'}), 400, headers
        
        logging.info(f"リクエスト内容: {json.dumps(request_json)}")
        
        # 必須パラメータの確認
        if 'prompt' not in request_json:
            logging.error("promptフィールドなし")
            return jsonify({'error': 'promptが必要です'}), 400, headers
        
        # 環境変数からプロジェクト情報を取得
        PROJECT_ID = os.environ.get('PROJECT_ID')
        LOCATION = os.environ.get('LOCATION', 'us-central1')
        
        logging.info(f"PROJECT_ID: {PROJECT_ID}, LOCATION: {LOCATION}")
        
        # モデル指定（リクエストから取得、デフォルトはgemini-pro）
        MODEL = request_json.get('model', 'gemini-2.0-flash-001')
        
        logging.info(f"使用モデル: {MODEL}")
        
        # APIエンドポイントの構築
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"
        
        logging.info(f"API URL: {url}")
        
        try:
            # 認証トークンの取得
            credentials, project_id = google.auth.default()
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            token = credentials.token
            
            logging.info("認証トークン取得成功")
            
            # Gemini APIにリクエストするヘッダー
            headers_for_gemini = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            # リクエストパラメータの取得（オプション）
            temperature = request_json.get('temperature', 0.7)
            max_output_tokens = request_json.get('max_output_tokens', 1024)
            top_p = request_json.get('top_p', 0.95)
            top_k = request_json.get('top_k', 40)
            
            # Gemini APIへのリクエストボディの構築
            gemini_request = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": request_json['prompt']
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens,
                    "topP": top_p,
                    "topK": top_k
                }
            }
            
            # システムインストラクション（オプション）
            if 'system_instruction' in request_json:
                gemini_request["systemInstruction"] = {
                    "parts": [
                        {
                            "text": request_json['system_instruction']
                        }
                    ]
                }
            
            logging.info(f"Gemini APIリクエスト: {json.dumps(gemini_request)}")
            
            # Gemini APIにリクエストを送信
            response = requests.post(url, headers=headers_for_gemini, json=gemini_request, timeout=30)
            
            logging.info(f"Gemini API レスポンスステータス: {response.status_code}")
            
            # レスポンスの確認と返却
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    logging.info("レスポンスJSONパース成功")
                    
                    # 応答テキストの抽出（実際のレスポンス構造に合わせて調整が必要）
                    try:
                        text = ""
                        if "candidates" in response_data and len(response_data["candidates"]) > 0:
                            candidate = response_data["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        text += part["text"]
                        
                        output = {
                            'response': response_data,
                            'text': text
                        }
                        
                        return jsonify(output), 200, headers
                    except Exception as e:
                        logging.error(f"テキスト抽出エラー: {str(e)}")
                        # レスポンス構造が想定と異なる場合でも、生のレスポンスを返す
                        return jsonify({
                            'response': response_data,
                            'error': f"テキスト抽出エラー: {str(e)}"
                        }), 200, headers
                        
                except json.JSONDecodeError as e:
                    logging.error(f"JSONデコードエラー: {str(e)}")
                    return jsonify({
                        'error': 'レスポンスのJSONパースに失敗しました',
                        'raw_response': response.text
                    }), 500, headers
            else:
                logging.error(f"API エラー: {response.status_code}, {response.text}")
                return jsonify({
                    'error': 'Gemini APIからのエラー',
                    'status_code': response.status_code,
                    'response': response.text
                }), response.status_code, headers
                
        except Exception as e:
            logging.error(f"API リクエスト中のエラー: {str(e)}")
            return jsonify({'error': f"API リクエスト中のエラー: {str(e)}"}), 500, headers
    
    except Exception as e:
        logging.error(f"全体的なエラー: {str(e)}")
        return jsonify({'error': str(e)}), 500, headers
        
# app = functions_framework.create_app(target='gemini_text_api')
