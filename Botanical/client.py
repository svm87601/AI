import requests

def ask_model(question):
    url = "http://10.1.88.115:5000/ask"
    payload = {"question": question}

    try:
        response = requests.post(url, json=payload, timeout=180)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        print("详细错误信息:", e)
        return {"error": str(e)}

if __name__ == "__main__":
    question = input("请输入你的问题: ")
    answer = ask_model(question)
    print("模型回答:", answer)


