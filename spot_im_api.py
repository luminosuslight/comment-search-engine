import requests
import re
import json
import pprint


def get_list_of_news_ids(year):
    sitemap_url = "https://www.rt.com/sitemap_%d.xml" % year
    articles_as_xml = requests.get(sitemap_url).text

    matches = re.findall(r'/(\d{6})-', articles_as_xml)

    print("Articles from %d: " % year, len(matches))

    return matches


def get_comments_of_news_as_json(news_id):
    # rt.com uses Spot.IM for the comment section
    # this message reads a JSON string directly from the corresponding Spot.IM page

    comment_url = "https://spoxy-shard5.spot.im/v2/spot/sp_6phY2k0C/post/%s/" % news_id
    html = requests.get(comment_url).text

    regex_for_json = re.compile(r'window.__APP_STATE__= JSON.parse\("(.*)"\)')
    regex_result = regex_for_json.search(html)
    json_raw = regex_result.group(1)
    json_raw = json_raw.replace('\\"', '"')
    json_raw = json_raw.replace('\\"', '"')

    print(json_raw[5000:5110])

    comments_in_spot_im_format = json.loads(json_raw)

    # print for debugging
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(comments_in_spot_im_format["conversations"][0]["conversation"]["comments"])
    print("--------------------------------")

    return comments_in_spot_im_format


def get_list_of_comments(comments_in_spot_im_format, url):
    comment_part = comments_in_spot_im_format["conversations"][0]["conversation"]["comments"]

    # comment_id, article_url, comment_author, comment_text, timestamp, parent_comment_id

    comments = []

    for root_comment in comment_part:
        comment = dict()
        comment["id"] = root_comment['id']
        comment["url"] = url
        comment["author"] = root_comment['user']['username']
        comment["text"] = root_comment['entities'][0]['text']
        comment["timestamp"] = root_comment['postedAt']
        comment["parent_id"] = ""
        comment["down_votes"] = root_comment['downVotesCount']
        comment["up_votes"] = root_comment['upVotesCount']
        comments.append(comment)

        for reply in root_comment['replies']:
            if reply.get('isDeleted', False):
                continue

            comment = dict()
            comment["id"] = reply['id']
            comment["url"] = url
            comment["author"] = reply['user']['username']
            comment["text"] = reply['entities'][0]['text']
            comment["timestamp"] = reply['postedAt']
            comment["parent_id"] = reply['rootMessageId']
            comment["down_votes"] = reply['downVotesCount']
            comment["up_votes"] = reply['upVotesCount']
            comments.append(comment)

        # print:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(root_comment)
        print("--------------------------------")

    return comments



if __name__ == '__main__':
    # news from 16 oct:
    # 'http://www.rt.com/usa/406881-trump-says-total-termination-of-iran-deal-possible/'
    # other news id that works: 407028
    #c = get_comments_of_news_as_json(406881)
    #comments = get_list_of_comments(c, "http://www.rt.com/usa/406881-trump-says-total-termination-of-iran-deal-possible/")

    #for com in comments:
    #    print(com)

    get_list_of_news_ids(2014)
