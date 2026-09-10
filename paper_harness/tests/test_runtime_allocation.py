from __future__ import annotations

import unittest

from harness.runtime_allocation import _datacenter


class RuntimeAllocationTests(unittest.TestCase):
    def test_datacenter_from_mast_host_formats(self) -> None:
        self.assertEqual(_datacenter("twshared43642.03.pci1"), "pci1")
        self.assertEqual(
            _datacenter("twshared43642.03.pci1.facebook.com"), "pci1"
        )
        self.assertEqual(
            _datacenter(
                "1b96-0551-1538-0a00.twshared43642.03.pci1.tw.fbinfra.net"
            ),
            "pci1",
        )

    def test_datacenter_rejects_host_without_location(self) -> None:
        with self.assertRaises(RuntimeError):
            _datacenter("localhost")


if __name__ == "__main__":
    unittest.main()
