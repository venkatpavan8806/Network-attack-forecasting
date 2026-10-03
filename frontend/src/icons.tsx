type Props = { size?: number; className?: string };

const base = (size = 20) => ({
  width: size,
  height: size,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
});

export const HomeIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M3 11.5 12 4l9 7.5" />
    <path d="M5 10v9a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1v-9" />
  </svg>
);

export const RadarIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <circle cx="12" cy="12" r="9" />
    <circle cx="12" cy="12" r="5" />
    <path d="M12 12 19 7" />
  </svg>
);

export const EyeIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

export const BarsIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M4 20V10" />
    <path d="M12 20V4" />
    <path d="M20 20v-7" />
  </svg>
);

export const NetworkIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <circle cx="5" cy="6" r="2.2" />
    <circle cx="19" cy="6" r="2.2" />
    <circle cx="12" cy="18" r="2.2" />
    <path d="M6.8 7.3 10.5 16.2M17.2 7.3 13.5 16.2M7.2 6h9.6" />
  </svg>
);

export const BellIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M6 9a6 6 0 0 1 12 0c0 4 1.5 5.5 1.5 5.5H4.5S6 13 6 9Z" />
    <path d="M10 19a2 2 0 0 0 4 0" />
  </svg>
);

export const LogIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <rect x="4" y="3" width="16" height="18" rx="2" />
    <path d="M8 8h8M8 12h8M8 16h5" />
  </svg>
);

export const ShieldIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M12 3 4.5 6v6c0 4.6 3.2 7.9 7.5 9 4.3-1.1 7.5-4.4 7.5-9V6L12 3Z" />
    <path d="m9 12 2 2 4-4" />
  </svg>
);

export const UploadIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M12 16V4M7 9l5-5 5 5" />
    <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
  </svg>
);

export const ArrowUpIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M12 19V5M6 11l6-6 6 6" />
  </svg>
);

export const ArrowDownIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M12 5v14M6 13l6 6 6-6" />
  </svg>
);

export const ClockIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3.5 2" />
  </svg>
);

export const AlertIcon = ({ size, className }: Props) => (
  <svg {...base(size)} className={className}>
    <path d="M10.3 3.9 1.9 18a1.5 1.5 0 0 0 1.3 2.2h17.6a1.5 1.5 0 0 0 1.3-2.2L13.7 3.9a1.5 1.5 0 0 0-2.6 0Z" />
    <path d="M12 9.5v4M12 16.5h.01" />
  </svg>
);
