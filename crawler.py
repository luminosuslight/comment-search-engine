import re
import json
import time
import csv
import os
import xml.etree.ElementTree as ET

import requests


def get_all_urls():
    # comment system (or at least the article IDs) in use since mid 2014
    # -> crawl 2014 - 2017
    urls = []
    for year in range(2014, 2018):
        urls += get_urls_of_year(year)
    return urls


def get_urls_of_year(year):
    sitemap_url = "https://www.rt.com/sitemap_%d.xml" % year
    try:
        articles_as_xml = requests.get(sitemap_url).text
    except Exception:
        return []
    root = ET.fromstring(articles_as_xml[38:])

    # all URLs start with https://www.rt.com/
    # we will remove it to reduce memory consumption
    prefix_length = len('https://www.rt.com/')

    urls = []
    for loc in root.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
        urls.append(loc.text[prefix_length:])
    print("Articles from %d: " % year, len(urls))
    return urls


def get_comments_of_article(url):
    # rt.com uses Spot.IM for the comment section
    # this method reads a JSON string directly from the corresponding Spot.IM page

    news_id = get_id_from_url(url)
    comment_url = "https://spoxy-shard5.spot.im/v2/spot/sp_6phY2k0C/post/%s/" % news_id
    try:
        html = requests.get(comment_url).text
    except Exception:
        return []

    # in the HTML file is a JSON part that includes the comments
    # this JSON part is extracted with a regex:
    regex_for_json = re.compile(r'window.__APP_STATE__= JSON.parse\("(.*)"\)')
    regex_result = regex_for_json.search(html)
    if not regex_result:
        return []
    json_raw = regex_result.group(1)
    json_raw = json_raw.replace('\\"', '"').replace('\\"', '"')

    # convert the JSON string to a Python object:
    comments_in_spot_im_format = json.loads(json_raw)
    comments = convert_spot_im_to_simple_comment_format(comments_in_spot_im_format, url)
    return comments


def get_id_from_url(url):
    match = re.match(r'.*/(\d{6})-', url)
    if not match:
        return None
    return match.group(1)


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


def crawl_and_save_year(year):
    urls_to_crawl = get_urls_of_year(year)

    comments = []
    urls_crawled = 0
    url_count = len(urls_to_crawl)
    sum_duration = 0
    try:
        for url in urls_to_crawl:
            try:
                begin = time.time()
                comments_of_article = get_comments_of_article(url)
                sum_duration += time.time() - begin
                # time.sleep(0.5)
                urls_crawled += 1
                if comments_of_article:
                    comments += comments_of_article
                    print("%d%% - Comments crawled: %d" % ((urls_crawled / url_count) * 100, len(comments)))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print("Error while crawling an article:", e)
                continue
    except KeyboardInterrupt:
        print("Stopped crawling articles.")

    print("Avg duration: %f" % (sum_duration / url_count))

    filename = "data/comments_%d.csv" % year
    save_comments_as_csv(comments, filename)
    print("Saved comments as '%s'." % filename)


def run():
    if not os.path.exists('data'):
        os.makedirs('data')

    crawl_and_save_year(2017)

    # Some Statistics:
    # avg. 302 bytes per comment
    # avg. 7.42 comments per article
    # ~96'000 articles
    # -> 215 MByte CSV file
    # avg. duration: 1.052442s / article
    # -> ~28h crawling runtime without additional wait times


if __name__ == '__main__':
    run()
