"""Safe public URL fetching and bounded uploads."""
import ipaddress, socket, urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from fastapi import HTTPException, UploadFile
from ..state import MAX_UPLOAD_BYTES
async def read_upload_limited(file: UploadFile, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read no more than limit+1 bytes so oversized uploads fail before decoding."""
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Upload exceeds the {limit // (1024 * 1024)} MB limit.")
    return data


class PinnedAddressAdapter(HTTPAdapter):
    """Connect to a validated IP while preserving the URL hostname for TLS and Host."""
    def __init__(self, hostname: str, address: str, port: int, scheme: str):
        super().__init__(max_retries=0)
        self.hostname, self.address, self.port, self.scheme = hostname, address, port, scheme

    def get_connection(self, url, proxies=None):
        if self.scheme == "https":
            return HTTPSConnectionPool(
                host=self.address, port=self.port, maxsize=1, block=True,
                assert_hostname=self.hostname, server_hostname=self.hostname,
            )
        return HTTPConnectionPool(host=self.address, port=self.port, maxsize=1, block=True)


def fetch_public_url(url: str):
    """Fetch a bounded public URL using the exact IP address checked before connecting."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use a public http:// or https:// URL.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError("The URL host could not be resolved.") from exc
    parsed_addresses = [ipaddress.ip_address(address) for address in addresses]
    if not parsed_addresses or any(not address.is_global for address in parsed_addresses):
        raise ValueError("The URL must resolve only to public addresses.")
    # Choose a resolved address and pin the socket to it; DNS cannot change between
    # validation and connection. TLS still verifies the original URL hostname.
    address = str(sorted(parsed_addresses, key=lambda item: (item.version, int(item)))[0])
    host_header = parsed.hostname if port in (80, 443) else f"{parsed.hostname}:{port}"
    if ":" in parsed.hostname:
        host_header = f"[{parsed.hostname}]" if port in (80, 443) else f"[{parsed.hostname}]:{port}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; TruthShield/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml,image/*;q=0.9,*/*;q=0.8",
        "Host": host_header,
    }
    response = requests.Response()
    with requests.Session() as session:
        session.trust_env = False
        session.mount("http://", PinnedAddressAdapter(parsed.hostname, address, port, parsed.scheme))
        session.mount("https://", PinnedAddressAdapter(parsed.hostname, address, port, parsed.scheme))
        response = session.get(url, headers=headers, timeout=10, allow_redirects=False, stream=True)
        if 300 <= response.status_code < 400:
            response.close()
            raise ValueError("Redirects are not followed. Enter the destination URL directly.")
        response.raise_for_status()
        chunks = []
        size = 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > 12 * 1024 * 1024:
                response.close()
                raise ValueError("The URL response is larger than the 12 MB limit.")
            chunks.append(chunk)
        response._content = b"".join(chunks)
        response.close()
    return response

