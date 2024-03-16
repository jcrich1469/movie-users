const path = require('path');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const webpack = require('webpack'); // Require the webpack module
const dotenv = require('dotenv').config(); // If you're using dotenv to manage environment variables

module.exports = {
  entry: './src/index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js',
  },
  devtool: 'source-map',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
        },
      },
    ],
  },
  resolve: {
    fallback: {
      "graceful-fs": false, // This correctly uses false to mock the module,
      "fs":false,
      "util": require.resolve("util/"), // Ensure this line ends with a comma if it's not the last in the list
      "assert": require.resolve("assert/"),
      "stream": require.resolve("stream-browserify"),
      "constants": require.resolve("constants-browserify"),
      "buffer": require.resolve("buffer/") // This is the last, so no comma is needed after this line
    },
  },
  plugins: [
    new webpack.DefinePlugin({
      'process.env.FIREBASE_API_KEY': JSON.stringify(process.env.FIREBASE_API_KEY), // Ensures FIREBASE_API_KEY is correctly injected
    }),
    new webpack.ProvidePlugin({
      process: 'process/browser', // Corrected syntax here
    }),
    new CleanWebpackPlugin()
  ]
};

