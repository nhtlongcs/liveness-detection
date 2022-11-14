import os
import os.path as osp
import urllib.request as urlreq


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    if osp.isfile(fpath):
        return fpath

    os.makedirs(root, exist_ok=True)

    try:
        print('Downloading ' + url + ' to ' + fpath)
        urlreq.urlretrieve(url, fpath)
    except (urlreq.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + fpath)
            urlreq.urlretrieve(url, fpath)

    return fpath
