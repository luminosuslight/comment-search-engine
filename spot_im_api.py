import requests
import re
import json
import pprint


# def get_rt_comments(article_url):
#     spotim_spotId = 'sp_6phY2k0C' # spotim id for RT
#     post_id = re.search('([0-9]+)', article_url).group(0)
#
#     # x = requests.post('https://api.spot.im/me/network-token/spotim')
#     # print(x)
#     # r1 = x.json()
#     spotim_token = "01171025gwZEnP"
#
#     payload = {
#         "count": 25, #number of comments to fetch
#         "sort_by":"best",
#         "cursor":{"offset":0,"comments_read":0},
#         "host_url": article_url,
#         "canonical_url": article_url
#     }
#
#     y = "https://open-api.spot.im/v1/spot-conversation-events/text-only?token=" + spotim_token + "&spot_id=" + spotim_spotId + "&post_id=406881&etag=0&count=2"
#
#     r2_url ='https://api.spot.im/conversation-read/spot/' + spotim_spotId + '/post/'+ post_id +'/get'
#     r2 = requests.post(r2_url, data=json.dumps(payload), headers={'X-Spotim-Token': spotim_token , "Content-Type": "application/json"})
#
#     print(requests.get(y).json())
#
#     print(r2)
#
#     return r2.json()


def get_comments_of_news_as_json(news_id):
    url = "https://spoxy-shard5.spot.im/v2/spot/sp_6phY2k0C/post/%s/" % news_id
    p = re.compile(r'window.__APP_STATE__= JSON.parse\("(.*)"\)')
    r = p.search(requests.get(url).text)
    j = r.group(1)
    j = j.replace('\\"', '"')
    obj = json.loads(j)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(obj["conversations"][0]["conversation"]["comments"])


if __name__ == '__main__':
    # news from 16 oct:
    # 'http://www.rt.com/usa/406881-trump-says-total-termination-of-iran-deal-possible/'
    # other news id that works: 407028
    get_comments_of_news_as_json(407028)
