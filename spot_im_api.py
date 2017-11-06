import requests
import re
import json
import pprint
import csv
import random


def get_ids_of_year(year):
    sitemap_url = "https://www.rt.com/sitemap_%d.xml" % year
    articles_as_xml = requests.get(sitemap_url).text
    matches = re.findall(r'/(\d{6})-', articles_as_xml)

    print("Articles from %d: " % year, len(matches))

    # FIXME: return URLs too!
    return matches


def get_all_ids():
    # comment system (or at least the article IDs) in use since mid 2014
    # -> crawl 2014 - 2017
    ids = []
    for year in range(2014, 2018):
        ids += get_ids_of_year(year)


def get_comments_in_spot_im_format(news_id):
    # rt.com uses Spot.IM for the comment section
    # this method reads a JSON string directly from the corresponding Spot.IM page

    comment_url = "https://spoxy-shard5.spot.im/v2/spot/sp_6phY2k0C/post/%s/" % news_id
    html = requests.get(comment_url).text

    # in the HTML file is a JSON part that includes the comments
    # this JSON part is extracted with a regex:
    regex_for_json = re.compile(r'window.__APP_STATE__= JSON.parse\("(.*)"\)')
    regex_result = regex_for_json.search(html)
    json_raw = regex_result.group(1)
    json_raw = json_raw.replace('\\"', '"').replace('\\"', '"')

    # convert the JSON string to a Python object:
    comments_in_spot_im_format = json.loads(json_raw)
    return comments_in_spot_im_format


def print_comments_for_debugging(news_id):
    comment_url = "https://spoxy-shard5.spot.im/v2/spot/sp_6phY2k0C/post/%s/" % news_id
    html = requests.get(comment_url).text

    regex_for_json = re.compile(r'window.__APP_STATE__= JSON.parse\("(.*)"\)')
    regex_result = regex_for_json.search(html)
    json_raw = regex_result.group(1)
    json_raw = json_raw.replace('\\"', '"')
    json_raw = json_raw.replace('\\"', '"')

    comments_in_spot_im_format = json.loads(json_raw)

    # print for debugging
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(comments_in_spot_im_format["conversations"][0]["conversation"]["comments"])
    print("--------------------------------")


def convert_spot_im_to_simple_comment_format(comments_in_spot_im_format, article_url):
    comment_part = comments_in_spot_im_format['conversations'][0]['conversation']['comments']

    comments = []
    for root_comment in comment_part:
        comment = dict()
        comment["id"] = root_comment['id']
        comment["url"] = article_url
        comment["author"] = root_comment['user']['username']
        comment["text"] = root_comment['entities'][0]['text']
        comment["timestamp"] = root_comment['postedAt']
        comment["parent_id"] = ''
        comment["up_votes"] = root_comment.get('upVotesCount', 0)
        comment["down_votes"] = root_comment.get('downVotesCount', 0)
        comments.append(comment)

        for reply in root_comment.get('replies', []):
            if reply.get('isDeleted', False):
                continue

            comment = dict()
            comment["id"] = reply['id']
            comment["url"] = article_url
            comment["author"] = reply['user']['username']
            comment["text"] = reply['entities'][0]['text']
            comment["timestamp"] = reply['postedAt']
            comment["parent_id"] = reply['rootMessageId']
            comment["up_votes"] = reply.get('upVotesCount', 0)
            comment["down_votes"] = reply.get('downVotesCount', 0)
            comments.append(comment)

    return comments


def save_comments_as_csv(comments, filename):
    fieldnames = ['id', 'url', 'author', 'text', 'timestamp', 'parent_id', 'up_votes', 'down_votes']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comments)


if __name__ == '__main__':

    ids = get_ids_of_year(2015)
    print(ids[:5])
    random_ids = random.sample(ids, 5)

    comments = []
    for news_id in random_ids:
        comments_in_spot_im_format = get_comments_in_spot_im_format(news_id)
        comments += convert_spot_im_to_simple_comment_format(comments_in_spot_im_format, "missing_url")

    len_sum = 0
    for c in comments:
        print(c)
    print("Comment count: ", len(comments))

    # save_comments_as_csv(comments, "comments.csv")
