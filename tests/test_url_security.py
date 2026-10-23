import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from truthshield.services import security
from truthshield.services.security import PinnedAddressAdapter, fetch_public_url


def dns_answer(address):
    return [(2, 1, 6, "", (address, 443))]


class TestPublicUrlFetchSecurity(unittest.TestCase):
    @patch("truthshield.services.security.requests.Session")
    @patch("truthshield.services.security.socket.getaddrinfo", return_value=dns_answer("127.0.0.1"))
    def test_rejects_private_addresses_before_connecting(self, _dns, session_factory):
        with self.assertRaisesRegex(ValueError, "public addresses"):
            fetch_public_url("https://example.test/article")
        session_factory.assert_not_called()

    @patch("truthshield.services.security.requests.Session")
    @patch("truthshield.services.security.socket.getaddrinfo", return_value=dns_answer("93.184.216.34") + dns_answer("10.0.0.8"))
    def test_rejects_mixed_public_and_private_dns_answers(self, _dns, session_factory):
        with self.assertRaisesRegex(ValueError, "public addresses"):
            fetch_public_url("https://example.test/article")
        session_factory.assert_not_called()

    @patch("truthshield.services.security.requests.Session")
    @patch("truthshield.services.security.socket.getaddrinfo", return_value=dns_answer("93.184.216.34"))
    def test_pins_connection_to_the_validated_address(self, _dns, session_factory):
        session = MagicMock()
        session_factory.return_value.__enter__.return_value = session
        response = MagicMock(status_code=200)
        response.iter_content.return_value = [b"page"]
        session.get.return_value = response

        result = fetch_public_url("https://example.test/article")

        adapters = [call.args[1] for call in session.mount.call_args_list]
        self.assertTrue(any(isinstance(adapter, PinnedAddressAdapter) and adapter.address == "93.184.216.34" for adapter in adapters))
        session.get.assert_called_once()
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])
        self.assertEqual(result._content, b"page")

    def test_https_adapter_does_not_resolve_hostname_again_after_validation(self):
        adapter = PinnedAddressAdapter("example.test", "93.184.216.34", 443, "https")
        with patch.object(security, "HTTPSConnectionPool") as pool_factory:
            adapter.get_connection("https://example.test/article")
        pool_factory.assert_called_once_with(
            host="93.184.216.34",
            port=443,
            maxsize=1,
            block=True,
            assert_hostname="example.test",
            server_hostname="example.test",
        )

    @patch("truthshield.services.security.requests.Session")
    @patch("truthshield.services.security.socket.getaddrinfo", return_value=dns_answer("93.184.216.34"))
    def test_rejects_redirect_responses(self, _dns, session_factory):
        session = MagicMock()
        session_factory.return_value.__enter__.return_value = session
        response = MagicMock(status_code=302)
        session.get.return_value = response

        with self.assertRaisesRegex(ValueError, "Redirects are not followed"):
            fetch_public_url("https://example.test/article")
        response.close.assert_called_once()

    @patch("truthshield.services.security.requests.Session")
    @patch("truthshield.services.security.socket.getaddrinfo", return_value=dns_answer("93.184.216.34"))
    def test_rejects_oversized_response(self, _dns, session_factory):
        session = MagicMock()
        session_factory.return_value.__enter__.return_value = session
        response = MagicMock(status_code=200)
        response.iter_content.return_value = [b"x" * (12 * 1024 * 1024), b"x"]
        session.get.return_value = response

        with self.assertRaisesRegex(ValueError, "larger than the 12 MB limit"):
            fetch_public_url("https://example.test/article")
        response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
