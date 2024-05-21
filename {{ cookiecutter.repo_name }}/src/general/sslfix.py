import ssl
import OpenSSL
import requests
import certifi


class FixSSL:
    """
    Checks & (if needed) downloads and adds SSL certifficate to certifi store
    """
    def __init__(self, host:str, port:int, custom_ca_path:str=None, verbose=True) -> None:
        self.host = host
        self.port = port
        self.custom_ca_path = custom_ca_path
        self.verbose = True
        self.certificate = None

    def check_host_access(self) -> bool:
        """
        tries to connect to the host, returns True in a case of success, False - otherwise
        """
        try:
            if self.verbose:
                print(f'Checking connection to {self.host}...')
            _ = requests.get(self.host)
            if self.verbose:
                print(f'Connection to {self.host} OK.')
            return True
        except requests.exceptions.SSLError as err:
            if self.verbose:
                print(f"Can't connect to {self.host}due to SSL Error:\n{err}")
            return False

    def fetch_host_certificate(self) -> None:
        """
        returns binary encoded certificate of a given host
        """
        # when ca file is provided add certificate from it
        if self.custom_ca_path:
            if self.verbose:
                print(f"fetching custom ca file from {self.custom_ca_path}...")
            self.certificate_encoded = self.certificate = self.read_host_certificate(self.custom_ca_path)
        # otherwise try to fetch and add it directly from the host
        else:
            server_host = self.host.replace("https://","")
            if self.verbose:
                print(f"fetching custom ca file from host: {server_host}")
            self.certificate = ssl.get_server_certificate((f"www.{server_host}", self.port)) #some servers may need following parameter - ssl_version=ssl.PROTOCOL_TLSv1
            self.certificate_encoded = str.encode(self.certificate)
            save_path = f"src/ssl/{server_host}_autosave.crt"
            self.save_host_certificate(self.certificate_encoded, save_path)
        if self.verbose:
            print("host certificate fetched!")

    def save_host_certificate(self, certificate: bytes, cert_filepath:str) -> None:
        """
        saves certificate from a provided file
        """
        with open(cert_filepath, 'wb') as outfile:
            outfile.write(certificate)

    def read_host_certificate(self, cert_filepath:str) -> bytes:
        """
        reads certificate from a provided file
        """
        with open(cert_filepath, 'rb') as infile:
            custom_ca_file = infile.read()
        return custom_ca_file

    def parse_certificate(self, certificate: str) -> dict:
        """
        supporting method: parses info from certificate
        """
        certificate_x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, certificate)
        certificate_info = dict()
        certificate_info['issuer'] = " ".join([k.decode("utf-8")+"="+v.decode("utf-8") for k,v in dict(certificate_x509.get_issuer().get_components()).items()])
        certificate_info['subject'] = " ".join([k.decode("utf-8")+"="+v.decode("utf-8") for k,v in dict(certificate_x509.get_subject().get_components()).items()])
        certificate_info['serial_num'] = certificate_x509.get_serial_number()
        certificate_info['SHA1_fp'] = certificate_x509.digest("sha1").decode("utf-8")
        certificate_info['SHA256_fp'] = certificate_x509.digest("sha256").decode("utf-8")
        certificate_info['MD5_fp'] = certificate_x509.digest("MD5").decode("utf-8")
        return certificate_info

    def fetch_certificate_info(self) -> None:
        """
        fetches certificates info into dictionary
        """
        if self.certificate == None:
            if self.verbose:
                print("Certificate is not fetched")
            self.get_host_certificate()
        self.certificate_info = self.parse_certificate(self.certificate)

    def build_cert_description(self, cert_info:dict) -> str:
        """
        supporting method: builds text description from certificate information dict
        """
        description = "\n\n# ADDED BY USER\n# Issuer:{}\n# Subject:{}\n# Serial:{}\n# SHA1 Fingerprint:{}\n# SHA256 Fingerprint:{}\n# MD5 Fingerprint:{}\n"\
                        .format(cert_info['issuer'], 
                                cert_info['subject'], 
                                cert_info['serial_num'], 
                                cert_info['SHA1_fp'], 
                                cert_info['SHA256_fp'], 
                                cert_info['MD5_fp'])
        description = str.encode(description)
        return description

    def run(self, custom_ca_path=None) -> None:
        """
        main method:
        """
        self.custom_ca_path = custom_ca_path
        if self.check_host_access():
            return True
        else:
            self.fetch_host_certificate()
            self.fetch_certificate_info()
            user_description = self.build_cert_description(self.certificate_info)

            ca_file = certifi.where()
            with open(ca_file, 'ab') as outfile:
                outfile.write(user_description)
                outfile.write(self.certificate_encoded)
            if self.verbose:
                print(f"Certificate for {self.host} was added to cafile")

if __name__ == "__main__":
    #TO DO: write some use case to test, etc
    cert_filepath = "src/ssl/hf_root.crt"
    url = 'https://drive.google.com/'
    port = 443

    fs = FixSSL(host=url, port=port, verbose=True)
    fs.run(cert_filepath)