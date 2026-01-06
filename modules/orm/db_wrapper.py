from sqlalchemy import create_engine
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

class DB_wrapper():
    """
    A wrapper for SQLAlchemy's create_engine function that allows for easy
    connection to a database. The wrapper also allows for easy configuration
    of SSL connections to the database.

    To interact with your specific database, you will need to install
    an additional pip package seperately. The following package are:
        PostgreSQL: psycopg2
        MySQL: pymysql
        MariaDB: pymysql
        MSSQL: pyodbc

    Attributes:
        engine (str): The database engine to use when connecting to the database.
            Must be one of: "postgresql", "mysql", "mariadb", or "mssql".
        database (str): The name of the database to connect to.
        username (str): The username to use when connecting to the database.
        password (str): The password to use when connecting to the database.
        host (str): The host to use when connecting to the database.
        port (int): The port to use when connecting to the database.
        ssl_mode (str): The SSL mode to use when connecting to the database.
            Must be one of: "none", "require", "verify-ca", "verify-full".
        ssl_ca_cert_path (str): The path to the SSL CA certificate.
        ssl_cert_path (str): The path to the SSL certificate.
        ssl_key_path (str): The path to the SSL key.
        url (str): The URL to use when connecting to the database.

    Methods:
        connect: Connects to the database and returns a connection object.
    """

    __DATABASE_ENGINES = {
        "postgresql": "postgresql+psycopg2",
        "mysql": "mysql+pymysql",
        "mariadb": "mysql+pymysql",
        "mssql": "mssql+pyodbc",
    }

    __SSL_MODES = {
        "none": "none",
        "require": "require",
        "verify-ca": "verify-ca",
        "verify-full": "verify-full",
    }

    def __init__(self,
        engine: str,
        database: str,
        host: str,
        port: int,
        username: str = None,
        password: str = None,
        ssl_mode: str = "none",
        ssl_ca_cert_path: str = None,
        ssl_cert_path: str = None,
        ssl_key_path: str = None
    ):
        self.engine = engine
        self.database = database
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.ssl_mode = ssl_mode
        self.ssl_ca_cert_path = ssl_ca_cert_path
        self.ssl_cert_path = ssl_cert_path
        self.ssl_key_path = ssl_key_path

    @property
    def engine(self) -> str:
        """
        Gets the database engine to use when connecting to the database.
        """
        return self.__engine

    @engine.setter
    def engine(self, engine: str) -> None:
        """
        Sets the database engine to use when connecting to the database.

        Args:
            engine (str): "postgresql", "mysql", "mariadb", or "mssql"

        """
        if not isinstance(engine, str):
            raise TypeError("Engine must be a string")

        if engine not in self.__DATABASE_ENGINES:
            raise ValueError(f"Engine must be one of: {', '.join(self.__DATABASE_ENGINES.keys())}")

        self.__engine = engine

    @property
    def database(self) -> str:
        """
        Gets the name of the database to connect to.
        """
        return self.__database

    @database.setter
    def database(self, database: str) -> None:
        """
        Sets the name of the database to connect to.
        """
        if not isinstance(database, str):
            raise TypeError("database must be a string")

        self.__database = database

    @property
    def host(self) -> str:
        """
        Gets the host to use when connecting to the database.
        """
        return self.__host

    @host.setter
    def host(self, host: str) -> None:
        """
        Sets the host to use when connecting to the database.
        """
        if not isinstance(host, str):
            raise TypeError("Host must be a string")

        self.__host = host

    @property
    def port(self) -> int:
        """
        Gets the port to use when connecting to the database.
        """
        return self.__port

    @port.setter
    def port(self, port: int) -> None:
        """
        Sets the port to use when connecting to the database.
        """
        if not isinstance(port, int):
            raise TypeError("Port must be an integer")

        self.__port = port

    @property
    def username(self) -> str:
        """
        Gets the username to use when connecting to the database.
        """
        return self.__username

    @username.setter
    def username(self, username: str) -> None:
        """
        Sets the username to use when connecting to the database.
        """
        if username is not None and not isinstance(username, str):
            raise TypeError("Username must be a string")

        self.__username = username

    @property
    def password(self) -> str:
        """
        Gets the password to use when connecting to the database.
        """
        return self.__password

    @password.setter
    def password(self, password: str) -> None:
        """
        Sets the password to use when connecting to the database.
        """
        if password is not None and not isinstance(password, str):
            raise TypeError("Password must be a string")

        self.__password = password

    @property
    def ssl_mode(self) -> str:
        """
        Gets the SSL mode to use when connecting to the database.
        """
        return self.__ssl_mode

    @ssl_mode.setter
    def ssl_mode(self, ssl_mode: str) -> None:
        """
        Sets the SSL mode to use when connecting to the database.

        Args:
            ssl_mode (str): "none", "require", "verify-ca", or "verify-full"
        """
        if not isinstance(ssl_mode, str):
            raise TypeError("SSL mode must be a string")

        if ssl_mode not in self.__SSL_MODES:
            raise ValueError("SSL mode must be one of: none, require, verify-ca, verify-full")

        self.__ssl_mode = ssl_mode

    @property
    def ssl_ca_cert_path(self) -> str:
        """
        Gets the path to the SSL CA certificate.
        """
        return self.__ssl_ca_cert_path

    @ssl_ca_cert_path.setter
    def ssl_ca_cert_path(self, ssl_ca_cert_path: str) -> None:
        """
        Sets the path to the SSL CA certificate.
        """
        if ssl_ca_cert_path is not None and not isinstance(ssl_ca_cert_path, str):
            raise TypeError("SSL CA cert path must be a string")

        self.__ssl_ca_cert_path = ssl_ca_cert_path

    @property
    def ssl_cert_path(self) -> str:
        """
        Gets the path to the SSL certificate.
        """
        return self.__ssl_cert_path

    @ssl_cert_path.setter
    def ssl_cert_path(self, ssl_cert_path: str) -> None:
        """
        Sets the path to the SSL certificate.
        """
        if ssl_cert_path is not None and not isinstance(ssl_cert_path, str):
            raise TypeError("SSL cert path must be a string")

        self.__ssl_cert_path = ssl_cert_path

    @property
    def ssl_key_path(self) -> str:
        """
        Gets the path to the SSL key.
        """
        return self.__ssl_key_path

    @ssl_key_path.setter
    def ssl_key_path(self, ssl_key_path: str) -> None:
        """
        Sets the path to the SSL key.
        """
        if ssl_key_path is not None and not isinstance(ssl_key_path, str):
            raise TypeError("SSL key path must be a string")

        self.__ssl_key_path = ssl_key_path

    @property
    def url(self) -> str:
        """
        Gets the URL to use when connecting to the database.
        """
        credentials = ""
        if self.username:
            credentials += self.username

        if self.username and self.password:
            credentials += f":{self.password}"

        if credentials:
            credentials += "@"

        return f"{self.__DATABASE_ENGINES[self.engine]}://{credentials}{self.host}:{self.port}/{self.database}{self.__ssl_config}"

    @property
    def __ssl_config(self) -> str:
        """
        Gets the SSL configuration to use when connecting to the database engine.
        """
        if self.engine == "postgresql":
            return self.__postgresql_ssl_config
        if self.engine == "mysql":
            return self.__mysql_ssl_config
        if self.engine == "maria":
            return self.__mariadb_ssl_config
        if self.engine == "mssql":
            return self.__mssql_ssl_config

    @property
    def __postgresql_ssl_config(self) -> str:
        """
        Gets the PostgreSQL SSL configuration to use when connecting to the database.
        """
        if self.ssl_mode == "none":
            return "?sslmode=disable"
        if self.ssl_mode == "require":
            return "?sslmode=require"
        if self.ssl_mode == "verify-ca":
            return f"?sslmode=verify-ca&sslrootcert={self.ssl_ca_cert_path}"
        if self.ssl_mode == "verify-full":
            return f"?sslmode=verify-full&sslrootcert={self.ssl_ca_cert_path}&sslcert={self.ssl_cert_path}&sslkey={self.ssl_key_path}"

    @property
    def __mysql_ssl_config(self) -> str:
        """
        Gets the MySQL SSL configuration to use when connecting to the database.
        """
        if self.ssl_mode == "none":
            return "?ssl=disabled"
        if self.ssl_mode == "require":
            return "?ssl=enabled"
        if self.ssl_mode == "verify-ca":
            return f"?ssl=enabled&ssl_ca={self.ssl_ca_cert_path}"
        if self.ssl_mode == "verify-full":
            return f"?ssl=enabled&ssl_ca={self.ssl_ca_cert_path}&ssl_cert={self.ssl_cert_path}&ssl_key={self.ssl_key_path}"

    @property
    def __mariadb_ssl_config(self) -> str:
        """
        Gets the MariaDB SSL configuration to use when connecting to the database.
        """
        return self.__mysql_ssl_config

    @property
    def __mssql_ssl_config(self) -> str:
        """
        Gets the MSSQL SSL configuration to use when connecting to the database.
        """
        if self.ssl_mode == "none":
            return "?driver=ODBC+Driver+17+for+SQL+Server"
        if self.ssl_mode == "require":
            return "?driver=ODBC+Driver+17+for+SQL+Server"
        if self.ssl_mode == "verify-ca":
            return f"?driver=ODBC+Driver+17+for+SQL+Server&sslca={self.ssl_ca_cert_path}"
        if self.ssl_mode == "verify-full":
            return f"?driver=ODBC+Driver+17+for+SQL+Server&sslca={self.ssl_ca_cert_path}&sslcert={self.ssl_cert_path}&sslkey={self.ssl_key_path}"

    @contextmanager
    def connect(self) -> Session:
        """
        Connects to the database and returns a connection object.

        Yields:
            connection: A connection object to the database.
        """
        session = sessionmaker(bind=create_engine(self.url))()
        yield session
        session.close()
