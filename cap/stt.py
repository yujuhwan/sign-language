import requests, json
import record as rc

rc.start()
rc.time.sleep(3)
rc.stop()

url = "https://kakaoi-newtone-openapi.kakao.com/v1/recognize"
key = '82e10401e10029b3aafa02767d3efc26'
headers = {
    "Content-Type": "application/octet-stream",
    # "Transfer-Encoding":"chunked",
    "Authorization": "KakaoAK " + key,
}

with open('record.wav', 'rb') as fp:
    audio = fp.read()

res = requests.post(url, headers=headers, data=audio)
# print(res.text) # 음성인식 결과 값 데이터 text파일

# result 데이터를 데이터 슬라이싱 하여 결과값만 호출
result_json_string = res.text[
                     res.text.index('{"type":"finalResult"'):res.text.rindex('}') + 1
                     ] # '{"type":"finalResult"' 이후 결과 값 슬라이싱

result = json.loads(result_json_string)
# print(result)
print(result['value'])