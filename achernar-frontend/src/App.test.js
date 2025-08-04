import { render, screen, waitFor } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      json: () => Promise.resolve({ message: 'Hello from API' }),
    }),
  );
});

afterEach(() => {
  jest.resetAllMocks();
});

test('renders dashboard heading and API message', async () => {
  render(<App />);

  expect(screen.getByText(/Achernar Dashboard \(MVP\)/i)).toBeInTheDocument();

  await waitFor(() =>
    expect(screen.getByText(/Hello from API/i)).toBeInTheDocument()
  );
});
