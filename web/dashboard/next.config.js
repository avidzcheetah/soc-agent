/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },
  async redirects() {
    return [
      {
        source: "/decisions",
        destination: "/",
        permanent: false,
      },
      {
        source: "/simulations",
        destination: "/",
        permanent: false,
      },
      {
        source: "/telemetry",
        destination: "/overview",
        permanent: false,
      },
      {
        source: "/incidents",
        destination: "/overview",
        permanent: false,
      },
      {
        source: "/research-results",
        destination: "/results",
        permanent: false,
      },
      {
        source: "/research",
        destination: "/results",
        permanent: false,
      },
      {
        source: "/evaluation",
        destination: "/results",
        permanent: false,
      },
    ];
  },
};

module.exports = nextConfig;
