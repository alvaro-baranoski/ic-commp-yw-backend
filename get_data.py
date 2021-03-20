import requests


def get_data_from_api(start_time, end_time, feed_id=506, interval=60, interval_type=1,
                      apikey='e9211b9101e5b2f654e7092611589d06', skip_missing=1, export=1):
    """
    Gets the information from the server API based on a number of paremeters included in the given URL
    Returns the data in a JSON like format
    -------------
    start_time: int, start time of the inverval of data given in milisseconds unix time
    end_time: int, end time of the inverval of data given in milisseconds unix time
    feed_id: int, optional, id of the feed present in the main server
    interval: int, optional, time inverval between points (can be seconds or samples/seconds)
    interval_type: bool, optional, selects the type of interval that the data will be given
                   0 = interval in seconds
                   1 = interval in samples/seconds
    apikey: string, optional, the read and write apikey presented on the server
    skip_missing: bool, optional, information about what will be done with missing data
                   0 = Puts NoneType on missing data points
                   1 = Ignores missing data points
    export: bool, optional, sets the data compression
                   0 = more data
                   1 = less data
    """

    data = requests.get(f"https://vega.eletrica.ufpr.br/emoncms/feed/data.json?"
                        f"id={feed_id}"
                        f"&start={start_time}"
                        f"&end={end_time}"
                        f"&interval={interval}"
                        f"&skipmissing={skip_missing}"
                        f"&apikey={apikey}"
                        f"&intervaltype={interval_type}"
                        f"&export={export}")

    return data.json()
