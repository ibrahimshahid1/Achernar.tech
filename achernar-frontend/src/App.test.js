import { render, screen } from '@testing-library/react';
import App from './App';

test('renders dashboard heading', () => {
  render(<App />);
  const headingElement = screen.getByRole('heading', { name: /Achernar Dashboard/i });
  expect(headingElement).toBeInTheDocument();
});
